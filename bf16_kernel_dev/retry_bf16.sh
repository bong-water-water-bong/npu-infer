#!/bin/bash
rm -rf /home/bcloud/.npu/cache/*
bash /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/run_iron_bf16_scalar.sh 2>&1
echo "Exit: $?"
