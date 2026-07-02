#!/bin/bash
set -e
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python"

echo "QKV..."
make M=256 K=1024 N=4096 m=128 k=64 n=128 use_placed=1 targetname=n2_core aie_py_src=n2_core_placed.py build/final_256x1024x4096_128x64x128.xclbin 2>&1 | tail -5

echo "O..."
make M=256 K=2048 N=1024 m=128 k=64 n=128 use_placed=1 targetname=n2_core aie_py_src=n2_core_placed.py build/final_256x2048x1024_128x64x128.xclbin 2>&1 | tail -5

echo "GU..."
make M=256 K=1024 N=6144 m=128 k=64 n=128 use_placed=1 targetname=n2_core aie_py_src=n2_core_placed.py build/final_256x1024x6144_128x64x128.xclbin 2>&1 | tail -5

echo "D..."
make M=256 K=3072 N=1024 m=128 k=64 n=128 use_placed=1 targetname=n2_core aie_py_src=n2_core_placed.py build/final_256x3072x1024_128x64x128.xclbin 2>&1 | tail -5

echo "Done"
ls -la build/final_256x*.xclbin 2>/dev/null
