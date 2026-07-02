#!/bin/bash
set -e
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
make M=256 K=1024 N=4096 m=128 k=64 n=128 use_placed=1 targetname=n2_core aie_py_src=n2_core_placed.py TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python" build/final_256x1024x4096_128x64x128.xclbin 2>&1 | tail -15
