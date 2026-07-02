#!/bin/bash
set -e
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python"

# Generate MLIR fresh
rm -f build/aie_256x1024x4096_128x64x128.mlir
echo "Generating fresh MLIR..."
/home/bcloud/torch2aie/.venv/bin/python n2_core_placed.py -M 256 -K 1024 -N 4096 -m 128 -k 64 -n 128 > build/aie_256x1024x4096_128x64x128.mlir
echo "MLIR: $(wc -l < build/aie_256x1024x4096_128x64x128.mlir) lines, $(grep -c 'for.*c16' build/aie_256x1024x4096_128x64x128.mlir) c16 loops"

# Build xclbin
echo "Building xclbin with standard aiecc..."
rm -rf build/aie_256x1024x4096_128x64x128.mlir.prj
aiecc --aietools="/home/bcloud/torch2aie/toolchain/aietools" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=final_256x1024x4096_128x64x128.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=insts_256x1024x4096_128x64x128.txt \
  build/aie_256x1024x4096_128x64x128.mlir 2>&1 | tail -10

echo "Instructions: $(wc -c < build/insts_256x1024x4096_128x64x128.txt) bytes"

# Test
echo "Testing..."
/home/bcloud/npu-sandbox/npu-infer/build/test_mt_gemm3 2>&1
