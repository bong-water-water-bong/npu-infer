#!/bin/bash
PYTHONPATH=/home/bcloud/torch2aie/toolchain/mlir_aie/python /home/bcloud/torch2aie/.venv/bin/python /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_i8.py -M 128 -K 1024 -N 4096 -m 128 -k 64 -n 128 2>/home/bcloud/npu-sandbox/npu-infer/build/int8/stderr.txt
echo "stdout size: $(wc -c < /home/bcloud/npu-sandbox/npu-infer/build/int8/QKV.mlir)"
echo "stderr:"
cat /home/bcloud/npu-sandbox/npu-infer/build/int8/stderr.txt
