#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"

# Step 1: Copy test kernel to qwen3_gemm dir
cp "$SCRIPT_DIR/mm_bf16_test.cc" "$NPU_INFER/build/qwen3_gemm/mm_bf16_direct.cc"

# Step 2: Compile test kernel with Chess compiler
TOOLCHAIN=/home/bcloud/torch2aie/toolchain
export PATH="$TOOLCHAIN/bin:$PATH"

cd "$NPU_INFER/build/qwen3_gemm"
xchesscc_wrapper aie2p \
  -c \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1 \
  mm_bf16_direct.cc \
  -o mm_bf16_direct.o 2>&1 | tee chess_build_test.log

echo "=== Test kernel compiled ==="
ls -la mm_bf16_direct.o
