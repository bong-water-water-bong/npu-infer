#!/bin/bash
# Create a proper INT8 n1_core MLIR generator from scratch
cat > /home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_i8.py << 'PYEOF'
import argparse
import numpy as np
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
import aie.utils.trace as trace_utils

def my_matmul(M, K, N, m, k, n):
    n_aie_rows = 1
    n_aie_cols = 8
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = m * k // 16
    
    tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 2 + n_aie_rows)]
    shim_tiles = tiles[0]
    mem_tiles = tiles[1]
    core_tiles = tiles[2:]
    
    A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l3l2_fifos = [None]
        A_l2l1_fifos = [None]
        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols
        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols
        
        shim_col = 0
        A_l3l2_fifos[0] = object_fifo(f"A_L3L2", shim_tiles[shim_col], mem_tiles[shim_col], 2, A_l2_ty, None, [[(m, k), (mtk // k, m * k), (k, 1)]])
        A_l2l1_fifos[0] = object_fifo(f"A_L2L1", mem_tiles[shim_col], core_tiles[0][0:n_aie_cols], 2, A_l1_ty, [(mtk // k * 4, m * k // 4), (k // 1, 1), (m // 4, k), (1, 1)], [[(k // 1, 1), (m // 4 // 1, 1), (1, 1)] for _ in range(n_aie_cols)])
        object_fifo_link(A_l3l2_fifos[0], A_l2l1_fifos[0])
        
        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty, [[(k, n), (k * n, 1), (1, 1)]], None)
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])
        
        for col in range(n_aie_cols):
            C_l1l2_fifos[0][col] = object_fifo(f"C_L1L2_{col}", core_tiles[0][col], mem_tiles[col], 2, C_l1_ty, None, [[(n, 1), (1, 1)]])
            C_l2l3_fifos[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty, [[(n, 1), (m, n), (1, 1)]], None)
            object_fifo_link(C_l1l2_fifos[0][col], C_l2l3_fifos[col])
        
        A_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, 1))
        B_taps = TensorTiler2D.group_tiler((K, N), (k, n), (1, 1))
        C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))
        
        num_col_tile = N // n // n_aie_cols
        num_groups = (M // m) * num_col_tile
        
        @core(core_tiles[0][0], stack_size=0xD00)
        def core_body():
            for _ in range_(8):
                for group in range_(num_groups):
                    for col in range(n_aie_cols):
                        A = A_l2l1_fifos[0].acquire(ObjectFifoPort.Consume, 1)
                        B = B_l2l1_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                        C = C_l1l2_fifos[0][col].acquire(ObjectFifoPort.Produce, 1)
                        matmul_scalar_i8_i16(A, B, C, M, K, N)
                        A_l2l1_fifos[0].release(ObjectFifoPort.Consume, 1)
                        B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                        C_l1l2_fifos[0][col].release(ObjectFifoPort.Produce, 1)

    my_matmul.device_body = device_body

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for p in ["M", "K", "N", "m", "k", "n"]:
        parser.add_argument(f"-{p}", type=int)
    args = parser.parse_args()
    my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
PYEOF
echo "Created INT8 generator"
