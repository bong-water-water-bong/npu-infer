#!/bin/bash
PYTHONPATH=/home/bcloud/torch2aie/toolchain/mlir_aie/python /home/bcloud/torch2aie/.venv/bin/python /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n2_core_placed.py -M 256 -K 1024 -N 4096 -m 128 -k 64 -n 128 > /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/test_n2.mlir 2>&1
wc -l /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/test_n2.mlir
head -5 /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/test_n2.mlir
