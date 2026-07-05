#!/bin/bash
# Build all INT8 xclbins for Qwen3-0.6B inference
# Uses n1_core_i8_v2.py generator with correct scalar kernel
set -e

export PATH=/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export MLIR_AIE_DIR=/home/bcloud/torch2aie/toolchain/mlir_aie

INT8_DIR=/home/bcloud/npu-sandbox/npu-infer/build/int8
GEN_DIR=/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev
mkdir -p "$INT8_DIR"

echo "=== Step 1: Compile INT8 scalar kernel ==="
xchesscc_wrapper aie2p -c \
  -I ${AIETOOLS_DIR}/include \
  -I ${MLIR_AIE_DIR}/include \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 \
  -I${MLIR_AIE_DIR}/include/aie_kernels \
  -Di8_i32_ONLY \
  ${MLIR_AIE_DIR}/include/aie_kernels/aie2p/mm.cc \
  -o "$INT8_DIR/mm_32x64x128.o" 2>&1 | tail -2
echo "Kernel .o: $(nm "$INT8_DIR/mm_32x64x128.o" | grep " T " | wc -l) functions"

echo ""
echo "=== Step 2: Build INT8 xclbins ==="

# (name  M      K      N    )
builds=(
  "KV"   "128"  "1024" "1024"
  "QKV"  "128"  "1024" "4096"
  "GU"   "128"  "1024" "6144"
  "O"    "128"  "2048" "1024"
  "D"    "128"  "3072" "1024"
)

for ((i=0; i<${#builds[@]}; i+=4)); do
  name="${builds[i]}"
  M="${builds[i+1]}"
  K="${builds[i+2]}"
  N="${builds[i+3]}"
  
  echo "  Building $name (${M}x${K}x${N})..."
  
  # Copy kernel .o (aiecc looks for mm_128x64x128.o in the prj dir)
  cp "$INT8_DIR/mm_32x64x128.o" "$INT8_DIR/mm_128x64x128.o"
  
  # Generate MLIR
  PYTHONPATH=/home/bcloud/torch2aie/toolchain/mlir_aie/python \
  /home/bcloud/torch2aie/.venv/bin/python \
    "$GEN_DIR/n1_core_i8_v2.py" \
    -M $M -K $K -N $N -m 32 -k 64 -n 128 > "$INT8_DIR/${name}_v2.mlir" 2>/dev/null
  
  # Build xclbin
  cd "$INT8_DIR"
  aiecc --aietools="$AIETOOLS_DIR" \
    --alloc-scheme=basic-sequential \
    --aie-generate-xclbin --no-compile-host \
    --xclbin-name="final_i8_${name}_v.xclbin" \
    --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --npu-insts-name="insts_i8_${name}_v.txt" \
    "${name}_v2.mlir" 2>&1 | tail -3
done

echo ""
echo "=== Build complete ==="
ls -la "$INT8_DIR"/final_i8_*_v.xclbin
