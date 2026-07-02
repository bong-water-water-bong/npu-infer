#!/bin/bash
# Build BF16 xclbin v2 by surgically editing the existing MLIR
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
TOOLCHAIN=/home/bcloud/torch2aie/toolchain
AIETOOLS=/home/bcloud/torch2aie/toolchain/aietools
export PATH="$TOOLCHAIN/bin:$PATH"
mkdir -p "$BUILD_DIR"

# Step 1: Copy kernel and MLIR source
cp "$SCRIPT_DIR/mm_bf16_v2.cc" "$BUILD_DIR/"
cp "$NPU_INFER/build/qwen3_gemm/design_1024_bf16_direct.mlir" "$BUILD_DIR/design_bf16_v2.mlir"

# Step 2: Compile kernel
echo "=== Compile BF16 v2 kernel ==="
cd "$BUILD_DIR"
xchesscc_wrapper aie2p \
  -c -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1 \
  -DDIM_M=128 -DDIM_K=64 -DDIM_N=128 \
  mm_bf16_v2.cc -o mm_bf16_v2.o 2>&1 | tee chess_v2.log
echo "  Kernel OK: $(stat -c%s mm_bf16_v2.o) bytes"

# Step 3: Surgically edit the MLIR for 64x64 B tiles and 128x64 C tiles
echo "=== Patch MLIR: 64x64 B tiles, 128x64 C tiles ==="
cd "$BUILD_DIR"

# B tile: 64x128 → 64x64
sed -i 's/memref<64x128xbf16>/memref<64x64xbf16>/g' design_bf16_v2.mlir

# C tile L1: 128x128 → 128x64
sed -i 's/memref<128x128xbf16>/memref<128x64xbf16>/g' design_bf16_v2.mlir

# C tile L3: 512x128 → 512x64
sed -i 's/memref<512x128xbf16>/memref<512x64xbf16>/g' design_bf16_v2.mlir

# A L3: 128x512 stays (it's the A buffer for rows×K-super-tile)
# A L1: 32x64 stays (it's the A sub-tile)

# Kernel names
sed -i 's/zero_kernel_bf16/zero_kernel_bf16_v2/g' design_bf16_v2.mlir
sed -i 's/matmul_vectorized_bf16/matmul_bf16_v2/g' design_bf16_v2.mlir
sed -i 's/mm_bf16_direct.o/mm_bf16_v2.o/g' design_bf16_v2.mlir

# Fix C output stream dimensions for 512x64 output
# Original: [size=64, stride=256, size=2, stride=8, size=16, stride=16, size=8, stride=1]
# Stride 256 was row_stride for 128-wide output (256 bf16 elements = 2 rows of 128)
# For 64-wide: row_stride = 64 elements
# Inner [16, 16] stays: 16 groups × 8 elements = full row width (128 for original)
# For 64-wide: [8, 16]: 8 groups × 8 elements = 64 = full row width
sed -i 's/<size = 64, stride = 256>, <size = 2, stride = 8>, <size = 16, stride = 16>, <size = 8, stride = 1>/<size = 64, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>/g' design_bf16_v2.mlir

# Fix link offsets for C L1→L3 (16384 = 128*128 bytes per tile → 8192 = 128*64 bytes)
sed -i 's/16384, 32768, 49152/8192, 16384, 24576/g' design_bf16_v2.mlir

echo "  MLIR lines: $(wc -l < design_bf16_v2.mlir)"
echo "  B tile refs: $(grep -c '64x64xbf16' design_bf16_v2.mlir)"
echo "  C tile refs: $(grep -c '128x64xbf16' design_bf16_v2.mlir)"

# Step 4: Build xclbin
echo "=== Build xclbin ==="
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

echo "=== Done ==="
ls -la design_bf16_v2.xclbin design_bf16_v2.insts 2>/dev/null || true
