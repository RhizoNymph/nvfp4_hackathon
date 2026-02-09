#!/usr/bin/env python3
"""
Profiling script for NVFP4 GEMM kernel with NCU.
This script runs the kernel once with minimal overhead for profiling.

Usage:
    # Basic NCU profiling (text output)
    ncu python profile_kernel.py

    # Full metrics with GUI-exportable report
    ncu --set full -o profile_report python profile_kernel.py

    # Specific metrics for memory/compute analysis
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed python profile_kernel.py
"""
import sys
import argparse
import torch

# Default benchmark case (you can change this)
DEFAULT_CASE = {"m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111}

# All benchmark cases from task.yml
BENCHMARK_CASES = {
    "small": {"m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
    "medium": {"m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    "large": {"m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
}


def profile_kernel(test_case: dict, warmup_runs: int = 2):
    """Run the kernel for profiling."""
    from reference import generate_input
    from submission import custom_kernel, compile_kernel
    
    m, n, k, l, seed = test_case["m"], test_case["n"], test_case["k"], test_case["l"], test_case["seed"]
    
    print(f"Profiling: m={m}, n={n}, k={k}, l={l}")
    print(f"Matrix sizes: A[{m}x{k}], B[{n}x{k}], C[{m}x{n}]")
    print(f"FLOPs: {2 * m * n * k * l / 1e9:.2f} GFLOPs")
    print()
    
    # Pre-compile the kernel (important: do this before NCU profiling region)
    print("Compiling kernel...", flush=True)
    compile_kernel()
    torch.cuda.synchronize()
    print("Compilation complete.")
    print()
    
    # Generate input
    print("Generating input data...", flush=True)
    data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
    torch.cuda.synchronize()
    print("Input generation complete.")
    print()
    
    # Warmup runs (these will be profiled too unless you use --launch-skip)
    if warmup_runs > 0:
        print(f"Running {warmup_runs} warmup iteration(s)...", flush=True)
        for i in range(warmup_runs):
            data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
            _ = custom_kernel(data_clone)
            torch.cuda.synchronize()
        print("Warmup complete.")
        print()
    
    # The actual kernel call that should be profiled
    print("=" * 50)
    print("RUNNING PROFILED KERNEL CALL")
    print("=" * 50)
    
    data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
    
    # Use CUDA range markers for easier identification in NCU
    torch.cuda.nvtx.range_push("nvfp4_gemm_kernel")
    output = custom_kernel(data_clone)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    print("Kernel execution complete.")
    print("=" * 50)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Profile NVFP4 GEMM kernel with NCU")
    parser.add_argument("--case", choices=list(BENCHMARK_CASES.keys()), default="large",
                        help="Which benchmark case to profile (default: large)")
    parser.add_argument("--m", type=int, help="Override M dimension")
    parser.add_argument("--n", type=int, help="Override N dimension")
    parser.add_argument("--k", type=int, help="Override K dimension")
    parser.add_argument("--l", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs (default: 2)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup runs")
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    print("NVFP4 GEMM Kernel Profiler")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Determine test case
    if args.m and args.n and args.k:
        test_case = {"m": args.m, "n": args.n, "k": args.k, "l": args.l, "seed": 1111}
    else:
        test_case = BENCHMARK_CASES[args.case]
    
    warmup = 0 if args.no_warmup else args.warmup
    profile_kernel(test_case, warmup_runs=warmup)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

