#!/bin/bash
set -e
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python"
# Try N=8192 with 40-col aiecc (has partition fix)
# First generate MLIR with standard toolchain
make M=128 K=1024 N=8192 m=128 k=64 n=128 use_placed=1 targetname=n1_core aie_py_src=n1_core_placed.py \
  build/aie_128x1024x8192_128x64x128.mlir 2>&1 | tail -5
echo "MLIR generated"
# Then compile with 40-col aiecc
export LD_LIBRARY_PATH="/home/bcloud/mlir-aie/npu2_40_toolchain/lib:$LD_LIBRARY_PATH"
/home/bcloud/mlir-aie/npu2_40_toolchain/bin/aiecc \
  --aietools="$TORCH2AIE_TOOLCHAIN/aietools" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=final_128x1024x8192_128x64x128.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=insts_128x1024x8192_128x64x128.txt \
  build/aie_128x1024x8192_128x64x128.mlir 2>&1 | tail -10
ls -la build/final_128x1024x8192_128x64x128.xclbin 2>/dev/null
