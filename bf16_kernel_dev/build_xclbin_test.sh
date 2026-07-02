#!/bin/bash
# Rebuild the BF16 direct xclbin with the test kernel
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"
MLIR_FILE="$NPU_INFER/build/qwen3_gemm/design_1024_bf16_direct.mlir"
WORK_DIR="$SCRIPT_DIR/build"
XCLBIN="$WORK_DIR/design_1024_bf16_test.xclbin"
INSTS="$WORK_DIR/design_1024_bf16_test.insts"

TOOLCHAIN=/home/bcloud/torch2aie/toolchain
export PATH="$TOOLCHAIN/bin:$PATH"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Copy the MLIR and kernel object
cp "$MLIR_FILE" design.mlir
cp "$NPU_INFER/build/qwen3_gemm/mm_bf16_direct.o" .

# Run aiecc to rebuild xclbin
# The compiler will automatically find mm_bf16_direct.o in the search path
aiecc \
  --aietools="$TOOLCHAIN/aietools" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin \
  --no-compile-host \
  --xclbin-name="$(basename $XCLBIN)" \
  --unified \
  --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name="$(basename $INSTS)" \
  design.mlir 2>&1 | tee aiecc_test.log

echo "=== XCLBIN built ==="
ls -la "$XCLBIN" 2>/dev/null || ls -la *.xclbin 2>/dev/null || echo "No xclbin found"
ls -la "$INSTS" 2>/dev/null || ls -la *.insts 2>/dev/null || echo "No insts found"
