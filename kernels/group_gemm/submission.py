import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

import functools
from typing import Tuple, List

import torch
from typing import TypedDict, TypeVar

input_t = TypeVar("input_t", bound=tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], list[tuple[int, int, int, int]]])
output_t = TypeVar("output_t", bound=list[torch.Tensor])

# Kernel configuration parameters
# Size of tma descriptor in bytes
bytes_per_tensormap = 128
# Number of tensormaps: a, b, sfa, sfb
num_tensormaps = 4
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (128, 128, 256)  
# Shape of the K dimension for the MMA instruction
mma_inst_shape_k = 64
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN  
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN  
# FP16 output type
c_dtype = cutlass.Float16  
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16  
# Number of threads per CUDA thread block
threads_per_cta = 192
# Stage numbers of shared memory and tmem
num_acc_stage = 1
num_ab_stage = 2
# Total number of columns in tmem
num_tmem_alloc_cols = 512
# Cluster shape (1x1 initially)
cluster_shape_mn = (1, 1)
# Max active clusters (B200 has 148 SMs)
max_active_clusters = 148


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tensor_of_abc_ptrs: cute.Tensor,
    tensor_of_sfasfb_ptrs: cute.Tensor,
    tensormaps: cute.Tensor,
    tensor_of_problem_sizes: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    cta_mn_list: List[Tuple[int, int]],
    num_tma_load_bytes: cutlass.Constexpr[int],
    cluster_layout_vmnk: cute.Layout,
):
    """
    Persistent GPU device kernel performing Group GEMM computation.
    CTAs loop over multiple tiles, enabling cross-tile pipelining.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()

    EPILOG_WARPS = (0,1,2,3)
    TMA_WARP = 4
    MMA_WARP = 5

    bidx, bidy, bidz = cute.arch.block_idx()
    gridDim_z = cute.arch.grid_dim()[2]

    # Total tiles across all groups (compile-time from cta_mn_list)
    total_tiles = sum(cta_m * cta_n for cta_m, cta_n in cta_mn_list)

    #
    # Define shared storage for kernel
    #
    size_tensormap_in_i64 = (
        num_tensormaps * bytes_per_tensormap // 8
    )
    @cute.struct
    class SharedStorage:
        tensormap_buffer: cute.struct.MemRange[
            cutlass.Int64, size_tensormap_in_i64
        ]
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        tmem_holding_buf: cutlass.Int32
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = (
        tensormap_a_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfa_smem_ptr = (
        tensormap_b_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfb_smem_ptr = (
        tensormap_sfa_smem_ptr
        + bytes_per_tensormap // 8
    )

    # Setup smem tensor for A, B, SFA, SFB
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    #
    # Initialize pipelines (persistent pattern - explicit state management)
    #
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(EPILOG_WARPS) * 32,
        ),
        cta_layout_vmnk=cluster_layout_vmnk,
    )

    # Cluster sync after barrier init
    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

    #
    # Local_tile partition global tensors (tile-independent, uses fake initial tensors)
    #
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )

    thr_mma = tiled_mma.get_slice(tidx)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)

    #
    # TMA partitions (tile-independent layout)
    #
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, 0, cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, 0, cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    # MMA fragments
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(
        cute.append(acc_shape, num_acc_stage)
    )

    # Tensormap manager and gmem pointers (per-CTA, indexed by bidz)
    tensormap_manager = utils.TensorMapManager(
        utils.TensorMapUpdateMode.SMEM, 128,
    )
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 0, None)].iterator
    )
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 1, None)].iterator
    )
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 2, None)].iterator
    )
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensormaps[(bidz, 3, None)].iterator
    )

    # Named barriers
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=32 * (1 + len(EPILOG_WARPS)),  # MMA + epilog warps
    )
    tensormap_ab_init_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=64,  # MMA warp (32) + TMA warp (32)
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )

    # Cluster wait before tensor memory alloc
    pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

    if warp_idx == TMA_WARP:
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage
        )

        tensormap_init_done = cutlass.Boolean(False)
        last_group_idx = cutlass.Int32(-1)

        tile_idx = bidz
        while tile_idx < total_tiles:
            # Delinearize tile_idx to group_idx, coord_x, coord_y
            group_idx = cutlass.Int32(0)
            find = False
            coord_x = cutlass.Int32(0)
            coord_y = cutlass.Int32(0)
            cta_rest = tile_idx
            for _, (cta_m, cta_n) in enumerate(cta_mn_list):
                if cta_rest >= (cta_m * cta_n):
                    group_idx += 1
                    cta_rest -= cta_m * cta_n
                else:
                    if not find:
                        coord_y = cta_rest // cta_m
                        coord_x = cta_rest % cta_m
                        cta_rest -= cta_m * cta_n
                        find = True

            is_group_changed = group_idx != last_group_idx

            # Update tensormap when group changes
            if is_group_changed:
                m = tensor_of_problem_sizes[group_idx, 0]
                n = tensor_of_problem_sizes[group_idx, 1]
                k = tensor_of_problem_sizes[group_idx, 2]
                l = tensor_of_problem_sizes[group_idx, 3]

                mA_mkl_iter = cute.make_ptr(
                    ab_dtype, tensor_of_abc_ptrs[group_idx, 0], cute.AddressSpace.gmem
                ).align(32)
                mB_nkl_iter = cute.make_ptr(
                    ab_dtype, tensor_of_abc_ptrs[group_idx, 1], cute.AddressSpace.gmem
                ).align(32)
                sfa_mkl_iter = cute.make_ptr(
                    sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 0], cute.AddressSpace.gmem
                ).align(32)
                sfb_nkl_iter = cute.make_ptr(
                    sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], cute.AddressSpace.gmem
                ).align(32)

                mA_mkl_layout = cute.make_layout(
                    (m, k, l), stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)))
                mB_nkl_layout = cute.make_layout(
                    (n, k, l), stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)))

                atom_shape = ((32, 4), (sf_vec_size, 4))
                atom_stride = ((16, 4), (0, 1))
                sfa_layout = cute.tile_to_shape(
                    cute.make_layout(atom_shape, stride=atom_stride),
                    mA_mkl_layout.shape, (2, 1, 3),
                )
                sfb_layout = cute.tile_to_shape(
                    cute.make_layout(atom_shape, stride=atom_stride),
                    mB_nkl_layout.shape, (2, 1, 3),
                )

                real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
                real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
                real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
                real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)

                if tensormap_init_done == False:
                    tensormap_ab_init_barrier.arrive_and_wait()
                    tensormap_init_done = True

                tensormap_manager.update_tensormap(
                    (real_tensor_a, real_tensor_b, real_tensor_sfa, real_tensor_sfb),
                    (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                    (tensormap_a_gmem_ptr, tensormap_b_gmem_ptr,
                     tensormap_sfa_gmem_ptr, tensormap_sfb_gmem_ptr),
                    TMA_WARP,
                    (tensormap_a_smem_ptr, tensormap_b_smem_ptr,
                     tensormap_sfa_smem_ptr, tensormap_sfb_smem_ptr),
                )

            # K tile count for current group
            cur_k = tensor_of_problem_sizes[group_idx, 2]
            k_tile_cnt = cute.ceil_div(cur_k, mma_tiler_mnk[2])

            # Slice global tensors to this tile's coords
            tAgA_slice = tAgA[(None, coord_x, None, 0)]
            tBgB_slice = tBgB[(None, coord_y, None, 0)]
            tAgSFA_slice = tAgSFA[(None, coord_x, None, 0)]
            tBgSFB_slice = tBgSFB[(None, coord_y, None, 0)]

            # Reset pipeline state for new tile
            ab_producer_state.reset_count()
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                    ab_producer_state
                )

            # Fence tensormap after update
            if is_group_changed:
                tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

            # K-loop: load tiles
            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.producer_acquire(
                    ab_producer_state, peek_ab_empty_status
                )

                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tensormap_a_gmem_ptr, cute.AddressSpace.generic),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tensormap_b_gmem_ptr, cute.AddressSpace.generic),
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_slice[(None, ab_producer_state.count)],
                    tAsSFA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tensormap_sfa_gmem_ptr, cute.AddressSpace.generic),
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB_slice[(None, ab_producer_state.count)],
                    tBsSFB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                        tensormap_sfb_gmem_ptr, cute.AddressSpace.generic),
                )

                ab_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

            tile_idx += gridDim_z
            last_group_idx = group_idx

        ab_pipeline.producer_tail(ab_producer_state)

    # =========================================================================
    # MMA WARP - Persistent tile loop for compute
    # =========================================================================
    if warp_idx == MMA_WARP:
        # Initialize tensormaps (overlaps with tmem allocation by epilog warp 0)
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_a, tensormap_a_smem_ptr, MMA_WARP
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_b, tensormap_b_smem_ptr, MMA_WARP
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfa, tensormap_sfa_smem_ptr, MMA_WARP
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfb, tensormap_sfb_smem_ptr, MMA_WARP
        )
        tensormap_ab_init_barrier.arrive_and_wait()

        # Wait for TMEM allocation
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        # SFA/SFB tmem tensors
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
            dtype=sf_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        # S2T copy partitions
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype,
        )
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
        )
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
        )
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        # Pipeline states
        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_ab_stage
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_acc_stage
        )

        tile_idx = bidz
        while tile_idx < total_tiles:
            # Delinearize
            group_idx = cutlass.Int32(0)
            find = False
            coord_x = cutlass.Int32(0)
            coord_y = cutlass.Int32(0)
            cta_rest = tile_idx
            for _, (cta_m, cta_n) in enumerate(cta_mn_list):
                if cta_rest >= (cta_m * cta_n):
                    group_idx += 1
                    cta_rest -= cta_m * cta_n
                else:
                    if not find:
                        coord_y = cta_rest // cta_m
                        coord_x = cta_rest % cta_m
                        cta_rest -= cta_m * cta_n
                        find = True

            cur_k = tensor_of_problem_sizes[group_idx, 2]
            k_tile_cnt = cute.ceil_div(cur_k, mma_tiler_mnk[2])

            # Get accumulator for this tile
            tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

            # Wait for accumulator buffer empty
            acc_pipeline.producer_acquire(acc_producer_state)

            # Reset consumer state for new tile
            ab_consumer_state.reset_count()
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(
                    ab_consumer_state
                )

            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.consumer_wait(
                    ab_consumer_state, peek_ab_full_status
                )

                # Copy SFA/SFB from shared memory to TMEM
                s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[s2t_stage_coord],
                    tCtSFB_compact_s2t,
                )

                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(
                        tcgen05.Field.SFA,
                        tCtSFA[sf_kblock_coord].iterator,
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB,
                        tCtSFB[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        tiled_mma, tCtAcc,
                        tCrA[kblock_coord], tCrB[kblock_coord],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

            acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            tile_idx += gridDim_z

        acc_pipeline.producer_tail(acc_producer_state)

    # =========================================================================
    # EPILOG WARPS - Persistent tile loop for output
    # =========================================================================
    if warp_idx in EPILOG_WARPS:
        # Allocate TMEM
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        # T2R copy setup (tile-independent)
        tCtAcc_s0 = tCtAcc_base[(None, None, None, 0)]
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_s0[None,0,0])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc_s0[None,0,0])

        # Register tensors (tile-independent shape)
        tDrAcc = cute.make_rmem_tensor(tDtAcc.shape, cutlass.Float32)
        tDrC = cute.make_rmem_tensor(tDtAcc.shape, c_dtype)

        # STG copy setup (tile-independent)
        simt_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16
        )
        thread_layout = cute.make_layout(
            (1, threads_per_cta), stride=(threads_per_cta, 1))
        value_layout = cute.make_layout((1, 1))
        tiled_copy_r2g = cute.make_tiled_copy_tv(
            simt_atom, thread_layout, value_layout
        )
        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

        # Pipeline state
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_acc_stage
        )

        tile_idx = bidz
        while tile_idx < total_tiles:
            # Delinearize
            group_idx = cutlass.Int32(0)
            find = False
            coord_x = cutlass.Int32(0)
            coord_y = cutlass.Int32(0)
            cta_rest = tile_idx
            for _, (cta_m, cta_n) in enumerate(cta_mn_list):
                if cta_rest >= (cta_m * cta_n):
                    group_idx += 1
                    cta_rest -= cta_m * cta_n
                else:
                    if not find:
                        coord_y = cta_rest // cta_m
                        coord_x = cta_rest % cta_m
                        cta_rest -= cta_m * cta_n
                        find = True

            # Construct C tensor for this group/tile
            m = tensor_of_problem_sizes[group_idx, 0]
            n = tensor_of_problem_sizes[group_idx, 1]
            l = tensor_of_problem_sizes[group_idx, 3]

            mC_mnl_iter = cute.make_ptr(
                c_dtype, tensor_of_abc_ptrs[group_idx, 2], cute.AddressSpace.gmem
            ).align(32)
            mC_mnl_layout = cute.make_layout(
                (m, n, l),
                stride=(cute.assume(n, 32), 1, cute.assume(m * n, 32)))
            mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)
            gC_mnl = cute.local_tile(
                mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)),
                (coord_x, coord_y, 0)
            )
            tCgC = thr_mma.partition_C(gC_mnl)
            tDgC = thr_copy_t2r.partition_D(tCgC[None,0,0])

            # Identity tensor and predicates
            cC = cute.make_identity_tensor(gC_mnl.shape)
            tDcC = thr_copy_r2g.partition_D(cC)
            tDpC = cute.make_rmem_tensor(tDrC.shape, cutlass.Boolean)
            residue_m = mC_mnl.shape[0] - cutlass.Int32(coord_x) * mma_tiler_mnk[0]
            residue_n = mC_mnl.shape[1] - cutlass.Int32(coord_y) * mma_tiler_mnk[1]
            for i in range(cute.size(tDrC.shape)):
                tDpC[i] = cute.elem_less(tDcC[i], (residue_n, residue_m))

            # Wait for accumulator buffer full
            acc_pipeline.consumer_wait(acc_consumer_state)

            # Copy accumulator to register, convert, store
            cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
            acc_vec = tDrAcc.load()
            tDrC.store(acc_vec.to(c_dtype))
            cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC),
                      pred=cute.flatten(tDpC))

            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()
            tile_idx += gridDim_z

        # Deallocate TMEM
        tmem.relinquish_alloc_permit()
        cute.arch.barrier()
        if warp_idx == EPILOG_WARPS[0]:
            tmem.free(acc_tmem_ptr, num_tmem_alloc_cols)
    pass


# Host-side JIT function to prepare tensors and launch GPU kernel.
@cute.jit
def my_kernel(
    ptr_of_tensor_of_problem_sizes: cute.Pointer,
    ptr_of_tensor_of_abc_ptrs: cute.Pointer,
    ptr_of_tensor_of_sfasfb_ptrs: cute.Pointer,
    ptr_of_tensor_of_tensormap: cute.Pointer,
    total_num_clusters: cutlass.Int32,
    problem_sizes: List[
        Tuple[int, int, int, int]
    ],  # Problem sizes for each group
    num_groups: cutlass.Int32,
):

    tensor_of_abc_ptrs = cute.make_tensor(
        ptr_of_tensor_of_abc_ptrs, cute.make_layout((num_groups, 3), stride=(3, 1))
    )
    tensor_of_sfasfb_ptrs = cute.make_tensor(
        ptr_of_tensor_of_sfasfb_ptrs, cute.make_layout((num_groups, 2), stride=(2, 1))
    )
    tensor_of_problem_sizes = cute.make_tensor(
        ptr_of_tensor_of_problem_sizes, cute.make_layout((num_groups, 4), stride=(4, 1))
    )
    grid_z = cutlass.min(total_num_clusters, max_active_clusters)
    tensor_of_tensormap = cute.make_tensor(
        ptr_of_tensor_of_tensormap, cute.make_layout((grid_z, 4, 16), stride=(64, 16, 1))
    )

    # Use fake shape for initial Tma descriptor and atom setup
    # The real Tma desc and atom will be updated during kernel execution.
    min_a_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    min_b_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    initial_a = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(
                cute.assume(min_a_shape[2], 32),
                1,
                cute.assume(min_a_shape[0] * min_a_shape[2], 32),
            ),
        ),
    )
    initial_b = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(
                cute.assume(min_b_shape[2], 32),
                1,
                cute.assume(min_b_shape[1] * min_b_shape[2], 32),
            ),
        ),
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_a.shape, sf_vec_size
    )
    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_b.shape, sf_vec_size
    )
    # Create initial SFA and SFB tensors with fake shape and null pointer.
    initial_sfa = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfa_layout)
    initial_sfb = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfb_layout)

    # Select MMA operation
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # Compute A/B/SFA/SFB/C shared memory layout
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    ) * atom_thr_size

    # Store CTA shape information for each Group in a List
    cta_mn_list = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes):
        x, y = cute.ceil_div(problem_sizes[group_idx][:2], mma_tiler_mnk[0:2])
        cta_mn_list.append((x, y))

    # Compute grid size - cap at max_active_clusters for persistent kernel
    grid_z = cutlass.min(total_num_clusters, max_active_clusters)
    grid = (1, 1, grid_z)

    # Launch the kernel
    kernel(
        tiled_mma,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa,
        tma_atom_sfb, tma_tensor_sfb,
        tensor_of_abc_ptrs,
        tensor_of_sfasfb_ptrs,
        tensor_of_tensormap,
        tensor_of_problem_sizes,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        cta_mn_list,
        num_tma_load_bytes,
        cluster_layout_vmnk,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(*cluster_shape_mn, 1),
    )
    return


# Global cache for compiled kernels (keyed by group size)
_compiled_kernel_cache = {}
# This function is used to compile the kernel once and cache it and then allow users to 
# run the kernel multiple times to get more accurate timing results.
def compile_kernel(problem_sizes):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache
    
    # Convert problem_sizes list to a hashable tuple for use as dictionary key
    cache_key = f"{len(problem_sizes)}"

    # Check if we already have a compiled kernel for these problem sizes
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    # Fake cluster numbers for compile only.
    total_num_clusters = cutlass.Int32(1)
    num_groups = cutlass.Int32(len(problem_sizes))
    # Each cluster needs its own set of tensormaps (one for A, B, SFA, SFB)
    # Shape: (total_num_clusters, num_tensormaps=4, bytes_per_tensormap/8=16)
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )
    compiled_func = cute.compile(
        my_kernel,
        cute_ptr_of_tensor_of_problem_sizes,
        cute_ptr_of_tensor_of_abc_ptrs,
        cute_ptr_of_tensor_of_sfasfb_ptrs,
        cute_ptr_of_tensor_of_tensormap,
        total_num_clusters,
        problem_sizes,
        num_groups
    )
    # Store compiled kernel in cache with problem_sizes as key
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.
    
    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.
    
    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            problem_sizes: list of tuples (m, n, k, l)
            each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
            l should always be 1 for each group.
            list size is the number of groups.
    
    Returns:
        list of c tensors where c is torch.Tensor[float16] of shape [m, n, l] for each group
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    compiled_func = compile_kernel(problem_sizes)

    # Extract raw data pointers from all input tensors for each group
    # These will be passed to the GPU kernel to access the actual tensor data
    abc_ptrs = []
    sfasfb_ptrs = []
    for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)):
        # Store pointers to A, B, and C matrices for this group
        abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
        # Store pointers to scale factor tensors for this group
        sfasfb_ptrs.append((sfa_reordered.data_ptr(), sfb_reordered.data_ptr()))

    # Create torch tensor to store problem sizes for all groups
    # Shape: (num_groups, 4) where each row contains (m, n, k, l) for that group
    # Layout: (num_groups, 4):(4, 1) means row-major storage
    tensor_of_problem_sizes = torch.tensor(
        problem_sizes, dtype=torch.int32, device="cuda"
    )

    # Create torch tensors to store data pointers for all groups
    # These allow the GPU kernel to dynamically access different tensors per group
    # tensor_of_abc_ptrs: Shape (num_groups, 3) containing (a_ptr, b_ptr, c_ptr) per group
    # tensor_of_sfasfb_ptrs: Shape (num_groups, 2) containing (sfa_ptr, sfb_ptr) per group
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
    tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

    # Compute the tile shape for each CUDA Thread Block (CTA)
    # cta_tile_shape_mn: [M_tile, N_tile] = [128, 128] for this kernel
    cta_tile_shape_mn = [128, mma_tiler_mnk[1]]
    # cluster_tile_shape_mn: Total tile shape per cluster (same as CTA since cluster is 1x1)
    cluster_tile_shape_mn = tuple(
        x * y for x, y in zip(cta_tile_shape_mn, (1, 1))
    )
    
    # Compute total number of cluster tiles needed across all groups
    # Each group's (m, n) dimensions are divided into tiles of size cluster_tile_shape_mn
    # This determines the total grid size (bidz dimension) for kernel launch
    total_num_clusters = 0
    num_groups = len(problem_sizes)
    for m, n, _, _ in problem_sizes:
        # Calculate number of tiles needed in M and N dimensions for this group
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        # Multiply M_tiles * N_tiles to get total tiles for this group
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    # Allocate device memory for tensormap descriptors
    # With persistent kernel, only grid_z CTAs are active (not total_num_clusters)
    grid_z = min(total_num_clusters, max_active_clusters)
    tensormap_shape = (
        grid_z,
        num_tensormaps,
        bytes_per_tensormap // 8,
    )
    tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")

    # Create CuTe pointers to the metadata tensors that will be passed to the kernel
    # These allow the GPU kernel to read problem sizes and tensor pointers
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_abc_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64,
        tensor_of_sfasfb_ptrs.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32,
        tensor_of_problem_sizes.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64,
        tensor_of_tensormap.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Launch the JIT-compiled GPU kernel with all prepared data
    # The kernel will perform block-scaled group GEMM: C = A * SFA * B * SFB for all groups
    compiled_func(
        cute_ptr_of_tensor_of_problem_sizes, # Pointer to problem sizes array
        cute_ptr_of_tensor_of_abc_ptrs,      # Pointer to ABC tensor pointers array
        cute_ptr_of_tensor_of_sfasfb_ptrs,   # Pointer to scale factor pointers array
        cute_ptr_of_tensor_of_tensormap,     # Pointer to tensormap buffer
        total_num_clusters,                  # Total number of CTAs to launch
        problem_sizes,                       # Problem sizes list (for host-side processing)
        num_groups,                          # Number of groups in this batch
    )

    res = []
    for i in range(num_groups):
        res.append(abc_tensors[i][2])
    return res