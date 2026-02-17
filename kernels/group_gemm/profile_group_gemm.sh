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

# Ensure uv venv is set up, then resolve the python path directly
# This avoids wrapping profilers around "uv run" which spawns child processes
# that ncu/nsys can't follow properly.
uv sync --quiet 2>/dev/null || true
PYTHON="$(uv run which python3)"
echo "Resolved Python: $PYTHON"

PREFIX="${KERNEL_TYPE}_case${CASE}"

echo "============================================================"
echo "NVFP4 Group GEMM Profiler"
echo "============================================================"
echo "  Kernel type : $KERNEL_TYPE"
echo "  Case        : $CASE"
echo "  Profiler    : $PROFILER"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Python      : $PYTHON"
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
        -t cuda,nvtx \
        --force-overwrite true \
        --stats false \
        "$PYTHON" profile_group_gemm.py $EXTRA_ARGS 2>&1 | tee "$NSYS_OUTPUT"

    echo ""
    if [[ -f "${NSYS_REP}.nsys-rep" ]]; then
        echo "Nsight Systems profiling complete!"
        echo "  Open ${NSYS_REP}.nsys-rep in Nsight Systems GUI for timeline analysis"
    else
        echo "WARNING: ${NSYS_REP}.nsys-rep was not generated!"
        echo "  Check the output above for errors."
    fi
    echo ""
}

# --- Nsight Compute ---
run_ncu() {
    local NCU_LOG="$OUTPUT_DIR/${PREFIX}_ncu_${TIMESTAMP}.txt"
    local NCU_REP="$OUTPUT_DIR/${PREFIX}_ncu_${TIMESTAMP}"
    local NCU_OUTPUT="$OUTPUT_DIR/${PREFIX}_ncu_output_${TIMESTAMP}.txt"

    echo "------------------------------------------------------------"
    echo "Running Nsight Compute..."
    echo "  Log    : $NCU_LOG"
    echo "  Report : ${NCU_REP}.ncu-rep"
    echo "  Output : $NCU_OUTPUT"
    echo "------------------------------------------------------------"
    echo ""

    ncu -o "$NCU_REP" \
        --log-file "$NCU_LOG" \
        --set roofline \
        --kernel-name-base demangled \
        --kernel-name regex:"cutlass.*" \
        --launch-count 1 \
        "$PYTHON" profile_group_gemm.py --no-warmup $EXTRA_ARGS 2>&1 | tee "$NCU_OUTPUT"

    echo ""
    if [[ -f "${NCU_REP}.ncu-rep" ]]; then
        echo "Nsight Compute profiling complete!"
        echo "  Check $NCU_LOG for metrics"
        echo "  Open ${NCU_REP}.ncu-rep in Nsight Compute GUI for interactive analysis"
    else
        echo "WARNING: ${NCU_REP}.ncu-rep was not generated!"
        echo "  Check the output above for errors."
    fi
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
