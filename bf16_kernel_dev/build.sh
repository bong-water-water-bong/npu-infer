#!/bin/bash
# Build BF16 native xclbin v2 — uses config1 Makefile infrastructure
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPU_INFER="$(cd "$SCRIPT_DIR/.." && pwd)"
TORCH2AIE="/home/bcloud/torch2aie"
CONFIG1="$TORCH2AIE/examples/gemm_asymmetric_tile_buffering/config1"
BUILD_DIR="$SCRIPT_DIR/build"
OUT_DIR="$NPU_INFER/build/qwen3_gemm"

mkdir -p "$BUILD_DIR"

# Step 1: Compile Chess kernel
echo "=== Compiling BF16 chess kernel v2 ==="
cp "$SCRIPT_DIR/mm_bf16_v2.cc" "$BUILD_DIR/mm_bf16_v2.cc"
cd "$BUILD_DIR"

# Use the same Chess compiler as the working config1 design
# The emulation flag enables BF16 mmul via BFP16 intrinsics
xchesscc_wrapper aie2p \
  -c \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1 \
  mm_bf16_v2.cc \
  -o mm_bf16_v2.o 2>&1 | tee chess_build.log

echo "Kernel compiled: mm_bf16_v2.o"

# Step 2: Generate MLIR design
echo "=== Generating MLIR design ==="

# Use the working MLIR design_1024_bfp16.mlir as a template
# Replace: bfp16 types → bf16, kernel name → matmul_bf16_v2
# Key changes from original:
#   A: 32×64 bf16 (same)
#   B: 64×128 bf16 (same as before, but depth=1 is confirmed OK at 16KB)
#   C: 128×128 bf16 (same)
#   Total: ~56KB + overhead — fits 64KB (confirmed via ELF analysis)

# Generate the MLIR
cat > "$BUILD_DIR/design_bf16_v2.mlir" << 'MLIR_EOF'
module {
  aie.device(npu2) {
    // BF16 kernel functions — linked from mm_bf16_v2.o
    func.func private @zero_kernel_bf16_v2(memref<128x128xbf16>)
      attributes {link_with = "mm_bf16_v2.o"}
    func.func private @matmul_bf16_v2(memref<32x64xbf16>, memref<64x128xbf16>, memref<128x128xbf16>)
      attributes {link_with = "mm_bf16_v2.o"}
MLIR_EOF

echo "MLIR stub generated. Run aiecc to build xclbin."
echo ""
echo "Next step (manual or via config1 Makefile):"
echo "  cd $BUILD_DIR"
echo "  aiecc --unified --dynamic-objFifos design_bf16_v2.mlir"
echo ""
echo "Or copy to config1/ and use:"
echo "  cp $BUILD_DIR/design_bf16_v2.mlir $CONFIG1/"
echo "  cd $CONFIG1 && make ..."
