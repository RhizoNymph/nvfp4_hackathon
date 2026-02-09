#!/usr/bin/env python3
"""
Profiling script for NVFP4 Group GEMM kernel with NCU and Nsight Systems.
This script runs the kernel once with minimal overhead for profiling.

Usage:
    # Profile with both NCU and Nsight Systems (default)
    ./profile_group_gemm.sh

    # Profile with only Nsight Compute
    ./profile_group_gemm.sh --profiler ncu

    # Profile with only Nsight Systems
    ./profile_group_gemm.sh --profiler nsys

    # Profile submission kernel instead
    ./profile_group_gemm.sh --submission

    # Basic NCU profiling (text output) - filter to only CUTLASS kernels
    ncu --kernel-name regex:"cutlass.*" python profile_group_gemm.py

    # Full metrics with GUI-exportable report
    ncu --set full --kernel-name regex:"cutlass.*" -o profile_report python profile_group_gemm.py

    # Nsight Systems profiling (timeline view)
    nsys profile -t cuda,nvtx -o profile_report python profile_group_gemm.py

    # Specific NCU metrics for memory/compute analysis
    ncu --kernel-name regex:"cutlass.*" --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed python profile_group_gemm.py

    # Profile specific benchmark case
    python profile_group_gemm.py --case 0

    # Profile with custom problem sizes
    python profile_group_gemm.py --m 128,256 --n 4096,4096 --k 7168,7168 --g 2
"""
import sys
import argparse
import torch

# Benchmark cases from task.yml
BENCHMARK_CASES = [
    {
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
        "k": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "g": 8,
        "seed": 1111,
        "name": "8 groups, small M, large N/K"
    },
    {
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168],
        "k": [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048],
        "g": 8,
        "seed": 1111,
        "name": "8 groups, varied M, large N, medium K"
    },
    {
        "m": [192, 320],
        "n": [3072, 3072],
        "k": [4096, 4096],
        "g": 2,
        "seed": 1111,
        "name": "2 groups, medium sizes"
    },
    {
        "m": [128, 384],
        "n": [4096, 4096],
        "k": [1536, 1536],
        "g": 2,
        "seed": 1111,
        "name": "2 groups, varied M, medium N/K"
    },
]

# Test cases from task.yml (smaller, for quick testing)
TEST_CASES = [
    {"m": [96, 128], "n": [128, 256], "k": [256, 512], "g": 2, "seed": 1111, "name": "test0"},
    {"m": [256, 72], "n": [512, 384], "k": [256, 256], "g": 2, "seed": 1111, "name": "test1"},
    {"m": [128, 128], "n": [128, 256], "k": [512, 256], "g": 2, "seed": 1111, "name": "test2"},
    {"m": [80, 128, 256], "n": [384, 256, 128], "k": [256, 512, 256], "g": 3, "seed": 1111, "name": "test3"},
]


def calculate_flops(m_list, n_list, k_list):
    """Calculate total FLOPs for the grouped GEMM."""
    total_flops = 0
    for m, n, k in zip(m_list, n_list, k_list):
        total_flops += 2 * m * n * k  # 2 ops per multiply-accumulate
    return total_flops


def profile_kernel(test_case: dict, warmup_runs: int = 2, use_submission: bool = False):
    """Run the kernel for profiling."""
    from reference import generate_input, ref_kernel

    m_list = test_case["m"]
    n_list = test_case["n"]
    k_list = test_case["k"]
    g = test_case["g"]
    seed = test_case["seed"]
    name = test_case.get("name", "custom")

    print(f"Profiling: {name}")
    print(f"Groups: {g}")
    print(f"Problem sizes:")
    for i in range(g):
        print(f"  Group {i}: M={m_list[i]}, N={n_list[i]}, K={k_list[i]}")

    total_flops = calculate_flops(m_list, n_list, k_list)
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print()

    # Load the kernel to use
    if use_submission:
        from submission import custom_kernel
        kernel_fn = custom_kernel
        kernel_name = "submission"
        print("Using SUBMISSION kernel")
    else:
        kernel_fn = ref_kernel
        kernel_name = "reference"
        print("Using REFERENCE kernel")
    print()

    # Generate input
    print("Generating input data...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    data = generate_input(m=m_list, n=n_list, k=k_list, g=g, seed=seed)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_delta = mem_after - mem_before
    print(f"Input generation complete. GPU memory: {mem_delta / 1e6:.1f} MB allocated ({mem_after / 1e6:.1f} MB total)")

    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
    print(f"  Problem sizes: {problem_sizes}")
    for i, (a, b, c) in enumerate(abc_tensors):
        print(f"  Group {i}: A {tuple(a.shape)} {a.dtype}, B {tuple(b.shape)} {b.dtype}, C {tuple(c.shape)} {c.dtype}")
    for i, (sfa, sfb) in enumerate(sfasfb_tensors):
        print(f"  Group {i} scale factors: SFA {tuple(sfa.shape)} {sfa.dtype}, SFB {tuple(sfb.shape)} {sfb.dtype}")
    print()

    # Clone function for data
    def clone_data(data):
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
        cloned_abc = [
            (a.clone(), b.clone(), c.clone())
            for a, b, c in abc_tensors
        ]
        cloned_sfasfb = [
            (sfa.clone(), sfb.clone())
            for sfa, sfb in sfasfb_tensors
        ]
        cloned_sfasfb_reordered = [
            (sfa.clone(), sfb.clone())
            for sfa, sfb in sfasfb_reordered_tensors
        ]
        return (cloned_abc, cloned_sfasfb, cloned_sfasfb_reordered, problem_sizes)

    # Warmup runs
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup iteration(s)...", flush=True)
        for i in range(warmup_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            data_clone = clone_data(data)
            start_event.record()
            _ = kernel_fn(data_clone)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            print(f"  Warmup {i+1}/{warmup_runs}: {elapsed_ms:.3f} ms", flush=True)
        print("Warmup complete.")
        print()

    # The actual kernel call that should be profiled
    print("=" * 50)
    print(f"RUNNING PROFILED KERNEL CALL ({kernel_name})")
    print("=" * 50)

    data_clone = clone_data(data)

    # Signal profiler to start capturing (used by nsys --capture-range=cudaProfilerApi)
    torch.cuda.cudart().cudaProfilerStart()
    # NVTX markers for easier identification in both NCU and nsys
    torch.cuda.nvtx.range_push(f"nvfp4_group_gemm_{kernel_name}")
    output = kernel_fn(data_clone)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    print("Kernel execution complete.")
    print("=" * 50)

    return output


def run_benchmark_timing(test_case: dict, num_runs: int = 10, use_submission: bool = False):
    """Run multiple iterations and measure timing."""
    from reference import generate_input, ref_kernel
    from utils import clear_l2_cache

    m_list = test_case["m"]
    n_list = test_case["n"]
    k_list = test_case["k"]
    g = test_case["g"]
    seed = test_case["seed"]
    name = test_case.get("name", "custom")

    # Load the kernel to use
    if use_submission:
        from submission import custom_kernel
        kernel_fn = custom_kernel
        kernel_name = "submission"
    else:
        kernel_fn = ref_kernel
        kernel_name = "reference"

    print(f"Benchmarking: {name} ({kernel_name})")
    print(f"Groups: {g}")
    for i in range(g):
        print(f"  Group {i}: M={m_list[i]}, N={n_list[i]}, K={k_list[i]}")

    total_flops = calculate_flops(m_list, n_list, k_list)
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")

    # Generate input
    print("Generating input data...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    data = generate_input(m=m_list, n=n_list, k=k_list, g=g, seed=seed)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    print(f"Input generation complete. GPU memory: {(mem_after - mem_before) / 1e6:.1f} MB allocated ({mem_after / 1e6:.1f} MB total)")

    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
    print(f"  Problem sizes: {problem_sizes}")
    for i, (a, b, c) in enumerate(abc_tensors):
        print(f"  Group {i}: A {tuple(a.shape)} {a.dtype}, B {tuple(b.shape)} {b.dtype}, C {tuple(c.shape)} {c.dtype}")
    print()

    def clone_data(data):
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data
        cloned_abc = [
            (a.clone(), b.clone(), c.clone())
            for a, b, c in abc_tensors
        ]
        cloned_sfasfb = [
            (sfa.clone(), sfb.clone())
            for sfa, sfb in sfasfb_tensors
        ]
        cloned_sfasfb_reordered = [
            (sfa.clone(), sfb.clone())
            for sfa, sfb in sfasfb_reordered_tensors
        ]
        return (cloned_abc, cloned_sfasfb, cloned_sfasfb_reordered, problem_sizes)

    # Warmup
    warmup_count = 3
    print(f"Running {warmup_count} warmup iteration(s)...", flush=True)
    for i in range(warmup_count):
        ws = torch.cuda.Event(enable_timing=True)
        we = torch.cuda.Event(enable_timing=True)
        ws.record()
        _ = kernel_fn(clone_data(data))
        we.record()
        torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup_count}: {ws.elapsed_time(we):.3f} ms", flush=True)
    print("Warmup complete.")
    print()

    # Timing runs
    print(f"Running {num_runs} benchmark iteration(s)...", flush=True)
    times_ms = []
    for i in range(num_runs):
        clear_l2_cache()
        data_clone = clone_data(data)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        _ = kernel_fn(data_clone)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)
        iter_tflops = total_flops / (elapsed_ms / 1000) / 1e12
        print(f"  Run {i+1}/{num_runs}: {elapsed_ms:.3f} ms ({iter_tflops:.2f} TFLOPS)", flush=True)

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    std_ms = (sum((t - avg_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

    tflops = total_flops / (avg_ms / 1000) / 1e12
    peak_tflops = total_flops / (min_ms / 1000) / 1e12

    print()
    print(f"  Avg:  {avg_ms:.3f} ms  ({tflops:.2f} TFLOPS)")
    print(f"  Min:  {min_ms:.3f} ms  ({peak_tflops:.2f} TFLOPS)")
    print(f"  Max:  {max_ms:.3f} ms")
    print(f"  Std:  {std_ms:.3f} ms  ({std_ms/avg_ms*100:.1f}% CV)")
    print()

    return avg_ms, tflops


def main():
    parser = argparse.ArgumentParser(description="Profile NVFP4 Group GEMM kernel")
    parser.add_argument("--case", type=int, default=0,
                        help="Benchmark case index (0-3, default: 0)")
    parser.add_argument("--test", type=int, default=None,
                        help="Use test case instead of benchmark (0-3)")
    parser.add_argument("--m", type=str, help="Override M dimensions (comma-separated)")
    parser.add_argument("--n", type=str, help="Override N dimensions (comma-separated)")
    parser.add_argument("--k", type=str, help="Override K dimensions (comma-separated)")
    parser.add_argument("--g", type=int, help="Number of groups (inferred from m/n/k if not set)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs (default: 2)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup runs")
    parser.add_argument("--submission", action="store_true", help="Profile submission kernel instead of reference")
    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark instead of single profiling run")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs (default: 10)")
    parser.add_argument("--all", action="store_true", help="Run all benchmark cases")
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    print("=" * 60)
    print("NVFP4 Group GEMM Kernel Profiler")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Determine test case(s)
    if args.all:
        # Run all benchmark cases
        print("Running all benchmark cases...")
        print()
        for i, case in enumerate(BENCHMARK_CASES):
            print(f"--- Benchmark Case {i} ---")
            if args.benchmark:
                run_benchmark_timing(case, args.runs, args.submission)
            else:
                profile_kernel(case, 0 if args.no_warmup else args.warmup, args.submission)
            print()
        return

    if args.m and args.n and args.k:
        # Custom problem sizes
        m_list = [int(x) for x in args.m.split(",")]
        n_list = [int(x) for x in args.n.split(",")]
        k_list = [int(x) for x in args.k.split(",")]
        g = args.g if args.g else len(m_list)
        test_case = {
            "m": m_list,
            "n": n_list,
            "k": k_list,
            "g": g,
            "seed": 1111,
            "name": "custom"
        }
    elif args.test is not None:
        # Use test case
        if args.test >= len(TEST_CASES):
            print(f"ERROR: Test case {args.test} not found (max: {len(TEST_CASES)-1})")
            sys.exit(1)
        test_case = TEST_CASES[args.test]
    else:
        # Use benchmark case
        if args.case >= len(BENCHMARK_CASES):
            print(f"ERROR: Benchmark case {args.case} not found (max: {len(BENCHMARK_CASES)-1})")
            sys.exit(1)
        test_case = BENCHMARK_CASES[args.case]

    warmup = 0 if args.no_warmup else args.warmup

    if args.benchmark:
        run_benchmark_timing(test_case, args.runs, args.submission)
    else:
        profile_kernel(test_case, warmup, args.submission)

    print("\nDone!")


if __name__ == "__main__":
    main()
