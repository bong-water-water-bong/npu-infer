#!/bin/bash
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
make M=128 K=4096 N=1024 m=128 k=64 n=128 use_placed=1 targetname=n1_core aie_py_src=n1_core_placed.py \
  TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python" \
  build/final_128x4096x1024_128x64x128.xclbin 2>&1 | tail -5
echo "O: $?"
