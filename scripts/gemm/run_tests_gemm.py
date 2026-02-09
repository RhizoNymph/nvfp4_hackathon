#!/usr/bin/env python3
"""
Simple test runner for the NVFP4 GEMM kernel.
Run from the nvfp4_gemm directory:
    python run_tests.py [--benchmark] [--test-only] [--benchmark-only]
"""
import sys
import time
import argparse
import torch

# Test cases from task.yml
TEST_CASES = [
    {"m": 128, "n": 256, "k": 256, "l": 1, "seed": 1111},
    {"m": 128, "n": 1536, "k": 7168, "l": 1, "seed": 1111},
    {"m": 128, "n": 3072, "k": 1536, "l": 1, "seed": 1111},
    {"m": 256, "n": 7168, "k": 256, "l": 1, "seed": 1111},
    {"m": 256, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
    {"m": 2304, "n": 4608, "k": 7168, "l": 1, "seed": 1111},
    {"m": 384, "n": 7168, "k": 2304, "l": 1, "seed": 1111},
    {"m": 512, "n": 512, "k": 7168, "l": 1, "seed": 1111},
    {"m": 512, "n": 4096, "k": 512, "l": 1, "seed": 1111},
    {"m": 512, "n": 1536, "k": 7168, "l": 1, "seed": 1111},
]

BENCHMARK_CASES = [
    {"m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
    {"m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
]

def run_single_test(test_case: dict, verbose: bool = True) -> tuple[bool, str, float]:
    """Run a single test case and return (passed, message, time_ms)."""
    from reference import generate_input, check_implementation
    from submission import custom_kernel
    
    m, n, k, l, seed = test_case["m"], test_case["n"], test_case["k"], test_case["l"], test_case["seed"]
    
    if verbose:
        print(f"  Testing m={m}, n={n}, k={k}, l={l}...", end=" ", flush=True)
    
    # Generate input
    data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
    
    # Clone for checking
    data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    try:
        output = custom_kernel(data_clone)
    except Exception as e:
        if verbose:
            print(f"FAILED (exception: {e})")
        return False, str(e), 0.0
    
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Check correctness
    passed, message = check_implementation(data, output)
    
    if verbose:
        if passed:
            print(f"PASSED ({elapsed_ms:.2f}ms)")
        else:
            print(f"FAILED: {message}")
    
    return passed, message, elapsed_ms


def run_benchmark(test_case: dict, num_warmup: int = 3, num_runs: int = 20) -> tuple[bool, float, float]:
    """Run benchmark and return (passed, mean_ms, std_ms)."""
    from reference import generate_input, check_implementation
    from submission import custom_kernel, compile_kernel
    
    m, n, k, l, seed = test_case["m"], test_case["n"], test_case["k"], test_case["l"], test_case["seed"]
    
    print(f"  Benchmarking m={m}, n={n}, k={k}, l={l}...")
    
    # Pre-compile the kernel
    print("    Compiling kernel...", end=" ", flush=True)
    compile_kernel()
    torch.cuda.synchronize()
    print("done")
    
    # Generate input
    data = generate_input(m=m, n=n, k=k, l=l, seed=seed)
    data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
    
    # Correctness check first
    output = custom_kernel(data_clone)
    torch.cuda.synchronize()
    passed, message = check_implementation(data, output)
    
    if not passed:
        print(f"    FAILED correctness check: {message}")
        return False, 0.0, 0.0
    
    # Warmup
    print(f"    Warming up ({num_warmup} runs)...", end=" ", flush=True)
    for _ in range(num_warmup):
        data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
        _ = custom_kernel(data_clone)
        torch.cuda.synchronize()
    print("done")
    
    # Benchmark
    times = []
    print(f"    Running benchmark ({num_runs} runs)...", end=" ", flush=True)
    for _ in range(num_runs):
        data_clone = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in data)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = custom_kernel(data_clone)
        end_event.record()
        
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / (len(times) - 1)) ** 0.5
    
    print(f"done")
    print(f"    Result: {mean_ms:.3f}ms ± {std_ms:.3f}ms (min: {min(times):.3f}ms, max: {max(times):.3f}ms)")
    
    return True, mean_ms, std_ms


def main():
    parser = argparse.ArgumentParser(description="Run NVFP4 GEMM kernel tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark cases instead of test cases")
    parser.add_argument("--test-only", action="store_true", help="Only run correctness tests (skip benchmarks)")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip correctness tests, run benchmarks directly")
    parser.add_argument("--single", type=int, help="Run only a single test case by index")
    args = parser.parse_args()
    
    # --benchmark-only implies --benchmark
    if args.benchmark_only:
        args.benchmark = True
    
    print("=" * 60)
    print("NVFP4 GEMM Kernel Test Runner")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    cases = BENCHMARK_CASES if args.benchmark else TEST_CASES
    
    if args.single is not None:
        if args.single >= len(cases):
            print(f"ERROR: Test index {args.single} out of range (0-{len(cases)-1})")
            sys.exit(1)
        cases = [cases[args.single]]
    
    # Skip correctness tests if --benchmark-only
    if not args.benchmark_only:
        # Run correctness tests
        print("Running correctness tests...")
        print("-" * 40)
        
        passed_count = 0
        failed_count = 0
        
        for case in cases:
            passed, message, elapsed = run_single_test(case)
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        
        print("-" * 40)
        print(f"Results: {passed_count} passed, {failed_count} failed")
        print()
        
        if failed_count > 0:
            print("Some tests failed! Skipping benchmarks.")
            sys.exit(1)
        
        if args.test_only:
            print("Test-only mode: skipping benchmarks.")
            sys.exit(0)
    else:
        print("Benchmark-only mode: skipping correctness tests.")
        print()
    
    # Run benchmarks
    if args.benchmark:
        print("Running benchmarks...")
        print("-" * 40)
        
        results = []
        for case in cases:
            passed, mean_ms, std_ms = run_benchmark(case)
            if passed:
                results.append((case, mean_ms, std_ms))
        
        print("-" * 40)
        print("\nBenchmark Summary:")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'L':>3} {'Mean (ms)':>12} {'Std (ms)':>12}")
        print("-" * 50)
        for case, mean_ms, std_ms in results:
            print(f"{case['m']:>6} {case['n']:>6} {case['k']:>6} {case['l']:>3} {mean_ms:>12.3f} {std_ms:>12.3f}")
        
        # Compute geometric mean
        if results:
            from math import prod
            geom_mean = prod(r[1] for r in results) ** (1/len(results))
            print("-" * 50)
            print(f"Geometric Mean: {geom_mean:.3f}ms")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

