#!/bin/bash
# Profile script that captures both ncu output and script output

OUTPUT_DIR="ncu_profiles"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCU_LOG="$OUTPUT_DIR/baseline_ncu_${TIMESTAMP}.txt"
NCU_REP="$OUTPUT_DIR/baseline_ncu_${TIMESTAMP}.ncu-rep"
ALL_OUTPUT="$OUTPUT_DIR/all_output_${TIMESTAMP}.txt"

echo "Profiling with Nsight Compute..."
echo "  - Nsight Compute log: $NCU_LOG"
echo "  - Binary report: $NCU_REP"
echo "  - Combined output: $ALL_OUTPUT"
echo ""

# Run ncu with log file, and also capture all output to combined file
ncu -o "$NCU_REP" \
    --log-file "$NCU_LOG" \
    --set full \
    uv run run_tests.py --benchmark-only 2>&1 | tee "$ALL_OUTPUT"

echo ""
echo "Profiling complete!"
echo "Check $NCU_LOG for Nsight Compute metrics"
echo "Check $ALL_OUTPUT for combined output"
echo "Open $NCU_REP in Nsight Compute GUI for interactive analysis"

