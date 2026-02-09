from torch._higher_order_ops.torchbind import call_torchbind_fake
import cuda.bindings.driver as cuda

import torch
from task import input_t, output_t

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

# Kernel configuration parameters
# Tile sizes for M, N, K dimensions
# M=128, N=128, K=256 are required for MmaMXF4NVF4Op with block-scaled GEMM
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
# Stage numbers for shared memory pipeline
num_acc_stage = 1
num_ab_stage = 5  # Optimal for all benchmark sizes
# Total number of columns in tmem
num_tmem_alloc_cols = 512

# Warp specialization IDs for persistent kernel
# Using 6 warps: 4 epilogue warps (0-3) + 1 MMA warp (4) + 1 TMA warp (5)
mma_warp_id = 4
tma_warp_id = 5
threads_per_cta = 32 * 6  # 192 threads total


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


# The CuTe persistent kernel for NVFP4 block-scaled GEMM with warp specialization
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
    mC_mnl: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    num_tma_load_bytes: cutlass.Constexpr[int],
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    """
    GPU device kernel performing persistent batched GEMM computation with warp specialization.
    """
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    #
    # Prefetch TMA descriptors (only TMA warp)
    #
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)

    #
    # Setup cta/thread coordinates
    #
    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)  # Always 0 for CtaGroup.ONE
    is_leader_cta = mma_tile_coord_v == 0  # Always true for CtaGroup.ONE
    tidx, _, _ = cute.arch.thread_idx()

    #
    # Define shared storage for kernel
    #
    @cute.struct
    class SharedStorage:
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    #
    # Initialize pipelines (with defer_sync=True for manual cluster sync)
    #
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )
    
    # AB pipeline - TMA warp produces, MMA warp consumes
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )

    # Accumulator pipeline - MMA warp produces, epilogue warps consume
    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    # 4 epilogue warps = warp 0,1,2,3
    num_epilog_warps = 4
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_epilog_warps
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )

    # Tensor memory allocation barrier (MMA warp + epilogue warps = 5 warps)
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=32 * 5,  # MMA warp + 4 epilogue warps
    )
    # Epilogue sync barrier (only epilogue warps = 4 warps)
    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=32 * 4,  # Only 4 epilogue warps
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=0,  # First epilogue warp allocates
    )

    # Cluster sync after barrier init
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

    #
    # Local_tile partition global tensors
    #
    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )
    k_tile_cnt = cute.size(gA_mkl, mode=[3])

    #
    # Partition global tensor for TiledMMA
    #
    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB = thr_mma.partition_B(gB_nkl)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCgC = thr_mma.partition_C(gC_mnl)

    #
    # TMA Partition for A/B/SFA/SFB
    #
    # TMA Partition_S/D for A
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    # TMA Partition_S/D for SFA
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # TMA Partition_S/D for SFB
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3),
        cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    #
    # Partition for TiledMMA A/B/C
    #
    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)
    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    #
    # Cluster wait before tensor memory alloc
    #
    pipeline_init_wait(cluster_shape_mn=(1, 1))

    # =====================================================================
    # SPECIALIZED TMA WARP - Handles all TMA loads
    # =====================================================================
    if warp_idx == tma_warp_id:
        # Create tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage
        )

        while work_tile.is_valid_tile:
            # Get tile coordinates from scheduler
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # Slice to per tile index
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
            tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgSFB_slice = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

            # Reset state for new tile
            ab_producer_state.reset_count()
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

            # TMA load loop
            for k_tile in range(k_tile_cnt):
                # Conditionally wait for AB buffer empty
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                # TMA load A/B/SFA/SFB
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA_slice[(None, ab_producer_state.count)],
                    tAsSFA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB_slice[(None, ab_producer_state.count)],
                    tBsSFB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )

                # Advance and peek next
                ab_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Wait for all buffers to be released
        ab_pipeline.producer_tail(ab_producer_state)

    # =====================================================================
    # SPECIALIZED MMA WARP - Handles all matrix multiply-accumulate
    # =====================================================================
    if warp_idx == mma_warp_id:
        # Wait for TMEM allocation
        tmem.wait_for_alloc()

        # Retrieve TMEM pointer and make accumulator tensor
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        # Make SFA/SFB tmem tensors
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        # Partition for S2T copy of SFA/SFB
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            sf_dtype,
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

        # Create tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_ab_stage
        )
        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_acc_stage
        )

        num_kblocks = cute.size(tCrA, mode=[2])

        while work_tile.is_valid_tile:
            # Reset state for new tile
            ab_consumer_state.reset_count()
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

            # Wait for accumulator buffer empty
            if is_leader_cta:
                acc_pipeline.producer_acquire(acc_producer_state)

            # Reset accumulate flag for new tile
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            # MMA mainloop
            for k_tile in range(k_tile_cnt):
                if is_leader_cta:
                    # Wait for AB buffer full
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                    # Copy SFA/SFB from smem to tmem
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

                    # MMA computation
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)

                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblock_coord],
                            tCrB[kblock_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # Release buffer
                    ab_pipeline.consumer_release(ab_consumer_state)

                # Advance and peek next (outside is_leader_cta check per reference example)
                ab_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    if is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

            # Commit accumulator
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Wait for accumulator buffer empty
        acc_pipeline.producer_tail(acc_producer_state)

    # =====================================================================
    # SPECIALIZED EPILOGUE WARPS (warp_idx < mma_warp_id) - Handle storing results
    # =====================================================================
    if warp_idx < mma_warp_id:
        # Allocate TMEM (only warp 0 actually allocates, but all participate in barrier)
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()

        # Retrieve TMEM pointer
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        # Partition for epilogue
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_base)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_base)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
        )
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
        )
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)

        # Create tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_acc_stage
        )

        while work_tile.is_valid_tile:
            # Get tile coordinates
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )

            # Wait for accumulator buffer full
            acc_pipeline.consumer_wait(acc_consumer_state)

            # Copy accumulator to register
            tTR_gC_slice = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]
            cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
            acc_vec = tTR_rAcc.load().to(c_dtype)
            tTR_rC.store(acc_vec)
            
            # Store C to global memory
            cute.copy(simt_atom, tTR_rC, tTR_gC_slice)

            # Release accumulator buffer (only one thread releases)
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Free TMEM
        tmem.relinquish_alloc_permit()
        epilog_sync_barrier.arrive_and_wait()
        tmem.free(acc_tmem_ptr)

    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, n, k, l = problem_size

    # Setup attributes that depend on gemm inputs
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n))
    )
    # Setup sfa/sfb tensor
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

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
        tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage
    )

    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        a_tensor,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        b_tensor,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfa_tensor,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB
    sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        sfb_tensor,
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

    # Compute grid size using persistent tile scheduler
    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )
    cluster_shape_mn = (1, 1)
    max_active_clusters = 148  # B200 has 148 SMs
    
    # Compute number of CTAs in M, N, L dimensions
    c_shape_mn = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
    gc = cute.zipped_divide(c_tensor, tiler=c_shape_mn)
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    cluster_shape_mnl = (*cluster_shape_mn, 1)
    
    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    # Launch the kernel
    kernel(
        tiled_mma,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa,
        tma_atom_sfb, tma_tensor_sfb,
        c_tensor,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        num_tma_load_bytes,
        tile_sched_params,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(*cluster_shape_mn, 1),
    )
    return


# Global cache for compiled kernel
_compiled_kernel = None

def compile_kernel():
    """Compile the kernel."""
    global _compiled_kernel
    
    if _compiled_kernel is not None:
        return _compiled_kernel
    
    # Create CuTe pointers
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    # Compile the kernel
    _compiled_kernel = cute.compile(my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0))
    
    return _compiled_kernel


def custom_kernel(data: input_t) -> output_t:
    """Execute the block-scaled GEMM kernel."""
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    # Get dimensions
    m, k, l = a.shape
    n, _, _ = b.shape
    k = k * 2  # Torch uses e2m1_x2 data type
    
    # Get compiled kernel
    compiled_func = compile_kernel()

    # Create CuTe pointers
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    # Execute the kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c
