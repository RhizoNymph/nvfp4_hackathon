#!/bin/bash
# Profile script for NVFP4 Group GEMM kernel
# Supports Nsight Compute (ncu) and Nsight Systems (nsys)

set -e

OUTPUT_DIR="profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default to reference kernel, can be overridden
KERNEL_TYPE="reference"
CASE="0"
PROFILER="both"  # ncu, nsys, or both
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profiler)
            PROFILER="$2"
            shift 2
            ;;
        --submission)
            KERNEL_TYPE="submission"
            EXTRA_ARGS="$EXTRA_ARGS --submission"
            shift
            ;;
        --case)
            CASE="$2"
            EXTRA_ARGS="$EXTRA_ARGS --case $2"
            shift 2
            ;;
        --test)
            CASE="test$2"
            EXTRA_ARGS="$EXTRA_ARGS --test $2"
            shift 2
            ;;
        --all)
            CASE="all"
            EXTRA_ARGS="$EXTRA_ARGS --all"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate profiler choice
if [[ "$PROFILER" != "ncu" && "$PROFILER" != "nsys" && "$PROFILER" != "both" ]]; then
    echo "ERROR: --profiler must be one of: ncu, nsys, both (got: $PROFILER)"
    exit 1
fi

PREFIX="${KERNEL_TYPE}_case${CASE}"

echo "============================================================"
echo "NVFP4 Group GEMM Profiler"
echo "============================================================"
echo "  Kernel type : $KERNEL_TYPE"
echo "  Case        : $CASE"
echo "  Profiler    : $PROFILER"
echo "  Output dir  : $OUTPUT_DIR"
echo ""

# --- Nsight Systems ---
run_nsys() {
    local NSYS_REP="$OUTPUT_DIR/${PREFIX}_nsys_${TIMESTAMP}"
    local NSYS_OUTPUT="$OUTPUT_DIR/${PREFIX}_nsys_output_${TIMESTAMP}.txt"

    echo "------------------------------------------------------------"
    echo "Running Nsight Systems..."
    echo "  Report : ${NSYS_REP}.nsys-rep"
    echo "  Output : $NSYS_OUTPUT"
    echo "------------------------------------------------------------"
    echo ""

    nsys profile \
        -o "$NSYS_REP" \
        -t cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite true \
        --stats true \
        --trace-fork-before-exec=true \
        uv run python3 profile_group_gemm.py $EXTRA_ARGS 2>&1 | tee "$NSYS_OUTPUT"

    echo ""
    echo "Nsight Systems profiling complete!"
    echo "  Open ${NSYS_REP}.nsys-rep in Nsight Systems GUI for timeline analysis"
    echo ""
}

# --- Nsight Compute ---
run_ncu() {
    local NCU_LOG="$OUTPUT_DIR/${PREFIX}_ncu_${TIMESTAMP}.txt"
    local NCU_REP="$OUTPUT_DIR/${PREFIX}_ncu_${TIMESTAMP}.ncu-rep"
    local NCU_OUTPUT="$OUTPUT_DIR/${PREFIX}_ncu_output_${TIMESTAMP}.txt"

    echo "------------------------------------------------------------"
    echo "Running Nsight Compute..."
    echo "  Log    : $NCU_LOG"
    echo "  Report : $NCU_REP"
    echo "  Output : $NCU_OUTPUT"
    echo "------------------------------------------------------------"
    echo ""

    ncu -o "$NCU_REP" \
        --log-file "$NCU_LOG" \
        --set roofline \
        --kernel-name-base demangled \
        --kernel-name regex:"cutlass.*" \
        --launch-count 1 \
        --profile-from-start off \
        --nvtx --nvtx-include "nvfp4_group_gemm_*" \
        --target-processes all \
        uv run python3 profile_group_gemm.py $EXTRA_ARGS 2>&1 | tee "$NCU_OUTPUT"

    echo ""
    echo "Nsight Compute profiling complete!"
    echo "  Check $NCU_LOG for metrics"
    echo "  Open $NCU_REP in Nsight Compute GUI for interactive analysis"
    echo ""
}

# --- Run selected profiler(s) ---
case "$PROFILER" in
    nsys)
        run_nsys
        ;;
    ncu)
        run_ncu
        ;;
    both)
        run_nsys
        run_ncu
        ;;
esac

echo "============================================================"
echo "All profiling complete!"
echo "============================================================"
