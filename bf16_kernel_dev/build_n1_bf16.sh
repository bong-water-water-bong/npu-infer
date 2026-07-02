#!/bin/bash
set -e
cd /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1
export PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH"
export PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python"
export TORCH2AIE_PYTHON="/home/bcloud/torch2aie/.venv/bin/python"

# Copy BF16 generator
cp /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n1_core_bf16.py .

# Step 1: Generate the MLIR (target name must match Makefile pattern)
echo "=== Generate BF16 n1 MLIR ==="
PYTHONPATH="/home/bcloud/torch2aie/toolchain/mlir_aie/python" /home/bcloud/torch2aie/.venv/bin/python n1_core_bf16.py -M 128 -K 1024 -N 2048 -m 128 -k 64 -n 128 > build/aie_bf16_n1.mlir 2>&1
echo "MLIR lines: $(wc -l < build/aie_bf16_n1.mlir)"

# Step 2: Compile kernel object for BF16 (identity: copy A to C)
echo "=== Compile identity kernel ==="
cat > /tmp/mm_bf16_id.cc << 'EOF'
#include <aie_api/aie.hpp>
extern "C" {
static int g_counter = 0;
void zero_kernel_bf16(bfloat16 *pC) {
  aie::vector<bfloat16, 64> z = aie::zeros<bfloat16,64>();
  for(int i=0;i<128*128;i+=64) aie::store_v(pC+i, z);
}
void matmul_vectorized_bf16(bfloat16 *pA, bfloat16 *pB, bfloat16 *pC) {
  bfloat16 *pCs = pC + g_counter * 32 * 128;
  if(g_counter == 3) g_counter = 0; else g_counter++;
  for(int i=0;i<32*64;i++) pCs[i] = pA[i];
}
}
EOF
xchesscc_wrapper aie2p -c -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1 /tmp/mm_bf16_id.cc -o build/mm_bf16_id.o 2>&1
echo "Kernel compiled: $(stat -c%s build/mm_bf16_id.o) bytes"

# Step 3: Compile xclbin with aiecc
echo "=== Compile xclbin ==="
aiecc --aietools="/home/bcloud/torch2aie/toolchain/aietools" \
  --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host \
  --xclbin-name=final_bf16_n1_test.xclbin \
  --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --npu-insts-name=insts_bf16_n1_test.txt \
  build/aie_bf16_n1.mlir 2>&1 | tail -10

ls -la build/final_bf16_n1_test.xclbin build/insts_bf16_n1_test.txt 2>/dev/null
