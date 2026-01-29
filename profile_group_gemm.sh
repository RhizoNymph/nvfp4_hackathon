#!/bin/bash
# Profile script for NVFP4 Group GEMM kernel
# Captures both NCU output and script output

set -e

OUTPUT_DIR="ncu_profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default to reference kernel, can be overridden
KERNEL_TYPE="reference"
CASE="0"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

NCU_LOG="$OUTPUT_DIR/${KERNEL_TYPE}_case${CASE}_ncu_${TIMESTAMP}.txt"
NCU_REP="$OUTPUT_DIR/${KERNEL_TYPE}_case${CASE}_ncu_${TIMESTAMP}.ncu-rep"
ALL_OUTPUT="$OUTPUT_DIR/${KERNEL_TYPE}_case${CASE}_output_${TIMESTAMP}.txt"

echo "Profiling NVFP4 Group GEMM with Nsight Compute..."
echo "  - Kernel type: $KERNEL_TYPE"
echo "  - Case: $CASE"
echo "  - NCU log: $NCU_LOG"
echo "  - Binary report: $NCU_REP"
echo "  - Combined output: $ALL_OUTPUT"
echo ""

# Run ncu with log file, and also capture all output to combined file
ncu -o "$NCU_REP" \
    --log-file "$NCU_LOG" \
    --set full \
    uv run profile_group_gemm.py $EXTRA_ARGS 2>&1 | tee "$ALL_OUTPUT"

echo ""
echo "Profiling complete!"
echo "Check $NCU_LOG for Nsight Compute metrics"
echo "Check $ALL_OUTPUT for combined output"
echo "Open $NCU_REP in Nsight Compute GUI for interactive analysis"
