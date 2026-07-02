#!/bin/bash
set -e
AIECC="/home/bcloud/mlir-aie/build/bin/aiecc"
export PATH="/home/bcloud/torch2aie/toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH"
export AIETOOLS_DIR="/home/bcloud/torch2aie/toolchain/aietools"
export LD_LIBRARY_PATH="/home/bcloud/mlir-aie/build/lib:/home/bcloud/torch2aie/toolchain/xrt/lib64"
export MLIR_AIE_DIR="/home/bcloud/torch2aie/toolchain/mlir_aie"
PEANO="/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie"
INT8_DIR="/home/bcloud/npu-sandbox/npu-infer/build/int8"
mkdir -p "$INT8_DIR"

# Step 1: Compile INT8 kernel with Peano (needed for --no-xchesscc flow)
echo "=== Compiling INT8 kernel with Peano ==="
PEANO_CLANG="$PEANO/bin/clang++"
"$PEANO_CLANG" \
  -c "/home/bcloud/mlir-aie/build/include/aie_kernels/aie2p/mm.cc" \
  -o "$INT8_DIR/mm_i8.o" \
  -DDIM_M=128 -DDIM_K=64 -DDIM_N=128 \
  -Di8_i16_ONLY -DSKIP_VECTORIZED \
  -I"/home/bcloud/mlir-aie/build/include" \
  -I"/home/bcloud/mlir-aie/build/include/aie_kernels" \
  -I"$AIETOOLS_DIR/include" \
  -I"$MLIR_AIE_DIR/include" \
  -std=c++20 -O2 -DNDEBUG \
  --target=aie2p-none-unknown-elf 2>&1 | tail -2
echo "Kernel: $(stat -c%s "$INT8_DIR/mm_i8.o") bytes"

# Step 2: Generate INT8 MLIR using n1_core_i8.py
echo "=== Generating INT8 MLIR ==="
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
/home/bcloud/torch2aie/.venv/bin/python \
  /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_i8.py \
  -M 128 -K 1024 -N 4096 -m 128 -k 64 -n 128 > "$INT8_DIR/QKV.mlir" 2>/dev/null
echo "MLIR: $(wc -l < "$INT8_DIR/QKV.mlir") lines"

# Fix MLIR kernel names
sed -i 's/link_with = "mm_128x64x128.o"/link_with = "mm_i8.o"/g' "$INT8_DIR/QKV.mlir"
sed -i 's/zero_kernel_bf16/zero_scalar_i16/g' "$INT8_DIR/QKV.mlir"
sed -i 's/matmul_vectorized_different_datatypes/matmul_scalar_i8_i16/g' "$INT8_DIR/QKV.mlir"

# Step 3: Build xclbin with patched aiecc (no --unified, link pre-compiled .o)
echo "=== Building INT8 xclbin ==="
cd "$INT8_DIR"
"$AIECC" \
  --aietools="$AIETOOLS_DIR" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name="final_i8_QKV.xclbin" \
  --no-xchesscc --no-xbridge \
  --peano="$PEANO" \
  --aie-generate-npu-insts \
  --npu-insts-name="insts_i8_QKV.txt" \
  "QKV.mlir" 2>&1 | tail -10

ls -la "final_i8_QKV.xclbin" 2>/dev/null && echo "✅ XCLBIN BUILT: $(stat -c%s final_i8_QKV.xclbin) bytes" || echo "❌ FAILED"
