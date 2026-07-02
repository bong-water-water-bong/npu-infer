#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/n1_core_12col.py" /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_12col.py
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/mlir-aie/npu2_40_toolchain/bin:$PATH"
export AIETOOLS_DIR="/home/bcloud/torch2aie/toolchain/aietools"
export TORCH2AIE_TOOLCHAIN="/home/bcloud/torch2aie/toolchain"
export LD_LIBRARY_PATH="/home/bcloud/mlir-aie/npu2_40_toolchain/lib:$LD_LIBRARY_PATH"
# Use 40-col aiecc
/home/bcloud/mlir-aie/npu2_40_toolchain/bin/aiecc \
  --aietools="$AIETOOLS_DIR" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=final_12col_test.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=insts_12col_test.txt \
  build/aie_128x1024x1536_128x64x128.mlir 2>&1 | tail -10
ls -la build/final_12col_test.xclbin build/insts_12col_test.txt 2>/dev/null
