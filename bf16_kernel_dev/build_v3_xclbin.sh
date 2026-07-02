#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
TOOLCHAIN=/home/bcloud/torch2aie/toolchain
AIETOOLS=/home/bcloud/torch2aie/toolchain/aietools
export PATH="$TOOLCHAIN/bin:$PATH"
mkdir -p "$BUILD_DIR"

# Copy v3 kernel (NO emulation flag) as mm_bf16_direct.o
cp "$NPU_INFER/build/qwen3_gemm/mm_bf16_v3.o" "$BUILD_DIR/mm_bf16_direct.o"

# Use the existing BF16 direct MLIR
cp "$NPU_INFER/build/qwen3_gemm/design_1024_bf16_direct.mlir" "$BUILD_DIR/design_bf16_v3.mlir"

# Fix kernel name in MLIR (v3 uses matmul_vectorized_bf16 which matches)
# The MLIR already uses matmul_vectorized_bf16 and zero_kernel_bf16
# Just change link_with to point to our new kernel
sed -i 's/mm_bf16_direct.o/mm_bf16_direct.o/g' "$BUILD_DIR/design_bf16_v3.mlir"

echo "Building xclbin with v3 kernel (no emulation)..."
cd "$BUILD_DIR"
aiecc \
  --aietools="$AIETOOLS" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=design_bf16_v3.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=design_bf16_v3.insts \
  design_bf16_v3.mlir 2>&1 | tail -10

echo "Done"
ls -la design_bf16_v3.xclbin design_bf16_v3.insts 2>/dev/null
