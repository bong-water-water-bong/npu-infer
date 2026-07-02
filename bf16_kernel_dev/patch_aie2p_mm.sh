#!/bin/bash
set -e
FILE="/home/bcloud/mlir-aie/build/include/aie_kernels/aie2p/mm.cc"
# Patch: add #ifndef SKIP_VECTORIZED around the vectorized combos line
sed -i '/^combos(matmul_vectorized_c_func) combos(zero_vectorized_c_func)/{
    i #ifndef SKIP_VECTORIZED
    a #endif
}' "$FILE"
grep -A2 -B2 "SKIP_VECTORIZED\|combos(matmul_vectorized" "$FILE"
