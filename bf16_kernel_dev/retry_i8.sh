#!/bin/bash
rm -rf /home/bcloud/.npu/cache/*
cmd="bash /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/run_iron_i8.sh"
$cmd > /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/iron_full.txt 2>&1
echo "Exit: $?"
wc -l /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/iron_full.txt
