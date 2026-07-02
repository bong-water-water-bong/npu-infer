#!/bin/bash
# Build fused xclbins: QKV (1024×4096) and GU (1024×6144)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1

TOOLCHAIN=/home/bcloud/torch2aie/toolchain
export PATH="$TOOLCHAIN/bin:$PATH"
export PYTHONPATH="$TOOLCHAIN/mlir_aie/python:$PYTHONPATH"

echo "=== Building fused QKV xclbin: 128×1024×4096 ==="
make M=128 K=1024 N=4096 m=128 k=64 n=128 use_placed=1 targetname=n1_core aie_py_src=n1_core_placed.py \
  build/final_128x1024x4096_128x64x128.xclbin 2>&1 | tail -20

echo ""
echo "=== Building fused GU xclbin: 128×1024×6144 ==="
make M=128 K=1024 N=6144 m=128 k=64 n=128 use_placed=1 targetname=n1_core aie_py_src=n1_core_placed.py \
  build/final_128x1024x6144_128x64x128.xclbin 2>&1 | tail -20

echo ""
echo "=== Done ==="
ls -la build/final_128x1024x4096_128x64x128.xclbin build/final_128x1024x6144_128x64x128.xclbin 2>/dev/null
