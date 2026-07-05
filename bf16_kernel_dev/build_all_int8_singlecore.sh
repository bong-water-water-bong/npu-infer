#!/bin/bash
# Build all INT8 xclbins using AMD's proven single_core.py IRON-API generator,
# replacing the broken hand-rolled n1_core_i8_v2.py generator.
#
# See docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md for the full investigation:
# n1_core_i8_v2.py's L2->L1 object_fifo calls for A and B never applied the
# r/s/t=8 AIE-mmul micro-tile reformatting (dims_to_stream) that single_core.py
# correctly does, and additionally routed data through an unnecessary
# intermediate mtk=512 L2 buffering layer not present in the proven design.
# Verified via isolated hardware GEMM test: all four shapes now match a
# software reference exactly (0 errors, 0 max diff) vs. essentially
# uncorrelated garbage before (mean error exceeding mean signal magnitude).
set -e

export PATH=/home/bcloud/mlir-aie/npu2_40_toolchain/bin:/home/bcloud/torch2aie/toolchain/aietools/bin:$PATH
export AIETOOLS_DIR=/home/bcloud/torch2aie/toolchain/aietools
export PYTHONPATH=/home/bcloud/mlir-aie/install/python

GEN=/home/bcloud/mlir-aie/programming_examples/basic/matrix_multiplication/single_core/single_core.py
PY=/home/bcloud/mlir-aie/.venv/bin/python3
OUT=/home/bcloud/npu-sandbox/npu-infer/build/int8
mkdir -p "$OUT"

# (name  M      K      N    )
builds=(
  "QKV"  "128"  "1024" "4096"
  "O"    "128"  "2048" "1024"
  "GU"   "128"  "1024" "6144"
  "D"    "128"  "3072" "1024"
)

for ((i=0; i<${#builds[@]}; i+=4)); do
  name="${builds[i]}"
  M="${builds[i+1]}"
  K="${builds[i+2]}"
  N="${builds[i+3]}"

  echo "=== Building $name (${M}x${K}x${N}) ==="
  "$PY" "$GEN" \
    -M "$M" -K "$K" -N "$N" -m 32 -k 64 -n 128 \
    --dtype_in i8 --dtype_out i32 --use-chess 1 \
    --xclbin-path "$OUT/final_i8_${name}_v.xclbin" \
    --insts-path "$OUT/insts_i8_${name}_v.txt" 2>&1 | tail -5
done

echo ""
echo "=== Build complete ==="
ls -la "$OUT"/final_i8_*_v.xclbin
