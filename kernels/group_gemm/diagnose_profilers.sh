#!/bin/bash
# Diagnose why ncu/nsys aren't producing binary output

echo "=== Tool Versions ==="
echo "--- ncu ---"
ncu --version 2>&1 || echo "ncu not found"
echo ""
echo "--- nsys ---"
nsys --version 2>&1 || echo "nsys not found"
echo ""
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>&1
echo ""

echo "=== Check for .qdstrm files (nsys intermediate) ==="
find profiles/ -name '*.qdstrm' -ls 2>/dev/null || echo "No .qdstrm files found"
echo ""

echo "=== Check for ANY ncu-rep or nsys-rep files ==="
find profiles/ -name '*.ncu-rep' -ls 2>/dev/null || echo "No .ncu-rep files"
find profiles/ -name '*.nsys-rep' -ls 2>/dev/null || echo "No .nsys-rep files"
echo ""

echo "=== Simple ncu test (profiling a basic matmul) ==="
PYTHON="$(uv run which python3 2>/dev/null || which python3)"
echo "Using python: $PYTHON"
echo ""

ncu --print-summary per-kernel \
    "$PYTHON" -c "
import torch
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()
print('matmul done')
" 2>&1

echo ""
echo "=== Simple nsys test ==="
nsys profile -o /tmp/nsys_diag_test --force-overwrite true \
    "$PYTHON" -c "
import torch
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()
print('matmul done')
" 2>&1

echo ""
if [[ -f /tmp/nsys_diag_test.nsys-rep ]]; then
    echo "nsys-rep generated OK: /tmp/nsys_diag_test.nsys-rep"
    ls -la /tmp/nsys_diag_test.nsys-rep
else
    echo "nsys-rep NOT generated"
    echo "Looking for intermediate files:"
    ls -la /tmp/nsys_diag_test* 2>/dev/null || echo "No files at all"
fi
echo ""

echo "=== Simple ncu -o test ==="
ncu -o /tmp/ncu_diag_test --set basic \
    "$PYTHON" -c "
import torch
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()
print('matmul done')
" 2>&1

echo ""
if [[ -f /tmp/ncu_diag_test.ncu-rep ]]; then
    echo "ncu-rep generated OK: /tmp/ncu_diag_test.ncu-rep"
    ls -la /tmp/ncu_diag_test.ncu-rep
else
    echo "ncu-rep NOT generated"
    ls -la /tmp/ncu_diag_test* 2>/dev/null || echo "No files at all"
fi
