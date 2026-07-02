#!/bin/bash
# Build the complete BF16 v2 xclbin: kernel → MLIR → aiecc → xclbin
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
TOOLCHAIN=/home/bcloud/torch2aie/toolchain
AIETOOLS=/home/bcloud/torch2aie/toolchain/aietools
export PATH="$TOOLCHAIN/bin:$PATH"
mkdir -p "$BUILD_DIR"

echo "=== Step 1: Compile BF16 v2 Chess kernel ==="
cp "$SCRIPT_DIR/mm_bf16_v2.cc" "$BUILD_DIR/mm_bf16_v2.cc"
cd "$BUILD_DIR"
xchesscc_wrapper aie2p \
  -c -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1 \
  mm_bf16_v2.cc -o mm_bf16_v2.o 2>&1 | tee chess_build.log
echo "Kernel compiled: $(ls -la mm_bf16_v2.o)"

echo "=== Step 2: Generate MLIR with 64x64 B tiles ==="
# Use the WORKING BFP16 MLIR as template, modify for BF16
# Key changes: B tile 64×64 bf16 (instead of 64×16 bfp16), C tile 128×64 (instead of 128×128)
# Depth=2 for B (2 × 8KB = 16KB), depth=2 for A (8KB), depth=1 for C (16KB)
# Total compute tile SRAM: 16+8+16 = 40KB + stack/code ≈ 45KB — FITS

python3 << 'PYEOF'
import re

# Read the working BFP16 MLIR as template
with open(f"{NPU_INFER}/build/qwen3_gemm/design_1024_bfp16.mlir") as f:
    mlir = f.read()

# Changes needed:
# 1. BFP16 type → bfloat16, and B tile from 64×16 → 64×64
# 2. C tile from 128×128 → 128×64
# 3. Kernel function names → mm_bf16_v2.o

mlir = mlir.replace(
    '!aiex.bfp<"v8bfp16ebs8">',  # BFP16 type
    'bfloat16'                     # BF16 type
)

# B tile L3/L2: memref<64x16xbfp16> → memref<64x64xbf16>
mlir = re.sub(
    r'memref<64x16xbf16>',
    'memref<64x64xbf16>',
    mlir
)

# B tile L1: same pattern
mlir = re.sub(
    r'memref<64x16xbfloat16>',
    'memref<64x64xbfloat16>',
    mlir
)

# C tile L1: memref<128x128xbf16> → memref<128x64xbf16>
mlir = re.sub(
    r'memref<128x128xbf16>',
    'memref<128x64xbf16>',
    mlir
)

# C tile L3/L2: memref<512x128xbf16> → memref<512x64xbf16>
mlir = re.sub(
    r'memref<512x128xbf16>',
    'memref<512x64xbf16>',
    mlir
)

# A tile L3: memref<128x512xbf16> stays same
# A tile L1: memref<32x64xbf16> stays same

# Kernel names: matmul_vectorized_different_datatypes → matmul_bf16_v2
mlir = mlir.replace(
    'matmul_vectorized_different_datatypes',
    'matmul_bf16_v2'
)
mlir = mlir.replace(
    'mm_128x64x128.o',
    'mm_bf16_v2.o'
)

with open(f"{BUILD_DIR}/design_bf16_v2.mlir", "w") as f:
    f.write(mlir)

print(f"Generated MLIR: {BUILD_DIR}/design_bf16_v2.mlir")
print(f"Lines: {len(mlir.splitlines())}")
PYEOF

echo "=== Step 3: Compile xclbin ==="
cd "$BUILD_DIR"
aiecc \
  --aietools="$AIETOOLS" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=design_bf16_v2.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=design_bf16_v2.insts \
  design_bf16_v2.mlir 2>&1 | tee aiecc_v2.log

echo "=== XCLBIN built ==="
ls -la design_bf16_v2.xclbin design_bf16_v2.insts 2>/dev/null
