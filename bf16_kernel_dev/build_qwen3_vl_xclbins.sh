#!/bin/bash
# Build INT8 xclbins for Qwen3-VL-4B (H=2560, NH=32, NKV=8, HD=128, IM=9728)
set -e

CFG1="/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1"
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export AIETOOLS_DIR="/home/bcloud/torch2aie/toolchain/aietools"
INT8_DIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"
mkdir -p "$INT8_DIR"
TAG="qwen3_vl_4b"

BUILD() {
    local label=$1 M=$2 K=$3 N=$4
    echo "=== Building ${label} ${M}x${K}x${N} ==="

    PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python" \
    /home/bcloud/torch2aie/.venv/bin/python "$CFG1/n1_core_i8.py" \
      -M $M -K $K -N $N -m 128 -k 64 -n 128 > "$INT8_DIR/$label.mlir" 2>/dev/null
    echo "  MLIR: $(wc -l < "$INT8_DIR/$label.mlir") lines"

    sed -i 's/zero_kernel_bf16/zero_scalar_i16/g' "$INT8_DIR/$label.mlir"
    sed -i 's/zero_kernel_bfloat16/zero_scalar_i16/g' "$INT8_DIR/$label.mlir"
    sed -i 's/link_with = "mm_128x64x128.o"/link_with = "mm_i8.o"/g' "$INT8_DIR/$label.mlir"

    cd "$INT8_DIR"
    aiecc --aietools="$AIETOOLS_DIR" \
      --alloc-scheme=basic-sequential \
      --aie-generate-xclbin --no-compile-host \
      --xclbin-name="final_i8_${label}_${TAG}.xclbin" \
      --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --npu-insts-name="insts_i8_${label}_${TAG}.txt" \
      "$label.mlir" 2>&1 | tail -5

    if [ -f "$INT8_DIR/final_i8_${label}_${TAG}.xclbin" ]; then
        echo "  ✅ $(stat -c%s "$INT8_DIR/final_i8_${label}_${TAG}.xclbin") bytes"
    else
        echo "  ❌ FAILED"
    fi
    echo ""
}

echo "=== Compiling INT8 kernel ==="
xchesscc_wrapper aie2p -c \
  "/home/bcloud/mlir-aie/build/include/aie_kernels/aie2p/mm.cc" \
  -o "$INT8_DIR/mm_i8.o" \
  -DDIM_M=128 -DDIM_K=64 -DDIM_N=128 \
  -Di8_i16_ONLY -DSKIP_VECTORIZED \
  -I"/home/bcloud/mlir-aie/build/include" \
  -I"/home/bcloud/mlir-aie/build/include/aie_kernels" \
  -I"$AIETOOLS_DIR/include" \
  -I"/home/bcloud/torch2aie/toolchain/mlir_aie/include" 2>&1 | grep -E "error|warning" | head -3
echo "Kernel compiled"
echo ""

# Qwen3-VL-4B dimensions:
# QKV: A[128,2560] × W[2560, 4096+1024+1024] = A[128,6144]
# O:   A[128,4096] × W[4096, 2560] = A[128,2560]
# G:   A[128,2560] × W[2560, 9728] = A[128,9728]
# U:   A[128,2560] × W[2560, 9728] = A[128,9728]
# D:   A[128,9728] × W[9728, 2560] = A[128,2560]

BUILD QKV 128 2560 6144
BUILD O   128 4096 2560
BUILD G   128 2560 9728
BUILD U   128 2560 9728
BUILD D   128 9728 2560

echo "=== All Qwen3-VL-4B INT8 xclbins ==="
ls -la "$INT8_DIR"/final_i8_*_${TAG}.xclbin
