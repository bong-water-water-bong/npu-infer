#!/bin/bash
# Build 2-layer batch INT8 xclbins
# Uses v2 MLIR generator (mm_32x64x128.o) with N doubled for 2-layer fusing
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"
GEN_V2="$SCRIPT_DIR/n1_core_i8_v2.py"

export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export AIETOOLS_DIR="/home/bcloud/torch2aie/toolchain/aietools"
mkdir -p "$BUILD_DIR"

BUILD() {
    local label=$1 M=$2 K=$3 N=$4
    echo "=== Building $label ${M}x${K}x${N} (2-layer) ==="
    
    # Generate MLIR
    source /home/bcloud/torch2aie/.venv/bin/activate
    PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python" \
    python "$GEN_V2" -M $M -K $K -N $N -m 32 -k 64 -n 128 \
      > "$BUILD_DIR/${label}_2layer.mlir" 2>/dev/null
    echo "  MLIR: $(wc -l < "$BUILD_DIR/${label}_2layer.mlir") lines"
    
    # Build xclbin
    cd "$BUILD_DIR"
    aiecc --aietools="$AIETOOLS_DIR" \
      --alloc-scheme=basic-sequential \
      --aie-generate-xclbin --no-compile-host \
      --xclbin-name="final_i8_${label}_2layer.xclbin" \
      --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --npu-insts-name="insts_i8_${label}_2layer.txt" \
      "${label}_2layer.mlir" 2>&1 | tail -5
    
    if [ -f "$BUILD_DIR/final_i8_${label}_2layer.xclbin" ]; then
        echo "  ✅ $(stat -c%s "$BUILD_DIR/final_i8_${label}_2layer.xclbin") bytes"
    else
        echo "  ❌ FAILED"
    fi
    deactivate
}

# Use the existing mm_32x64x128.o kernel — already compiled
echo "Kernel: $BUILD_DIR/mm_32x64x128.o"
ls -la "$BUILD_DIR/mm_32x64x128.o"

# Build 2-layer projections
# QKV: 2×4096 = 8192 (both layers' QKV fused)
BUILD QKV 128 1024 8192

# O: 2×1024 = 2048 (both layers' O fused)
BUILD O 128 2048 2048

# GU: 2×6144 = 12288 (both layers' GU fused)
BUILD GU 128 1024 12288

# D: 2×1024 = 2048 (both layers' D fused)
BUILD D 128 3072 2048

echo ""
echo "=== 2-layer batch xclbins ==="
ls -la "$BUILD_DIR"/final_i8_*_2layer.xclbin 2>/dev/null

# Check instruction counts
for f in "$BUILD_DIR"/insts_i8_*_2layer.txt; do
    if [ -f "$f" ]; then
        words=$(wc -w < "$f")
        echo "  $(basename $f): $words instructions"
    fi
done
