#!/bin/bash
set -e
cp /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/mm_bf16_v3.cc /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/mm_bf16_v3.cc
cd /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm
PATH="/home/bcloud/torch2aie/toolchain/bin:$PATH" xchesscc_wrapper aie2p -c -DDIM_M=128 -DDIM_K=64 -DDIM_N=128 mm_bf16_v3.cc -o mm_bf16_v3.o 2>&1
echo "exit: $?"
ls -la mm_bf16_v3.o
