#!/usr/bin/env python3
#
# INT8 MLIR generator using SCALAR kernel for flat access
# No stride patterns, no vectorized MMUL
# Each call processes 32x64x128
# The loop structure handles M-tiling (4 acquires = 128 M rows)
# and K-tiling (16 iterations = 1024 K)
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=32)  # <-- CHANGED to 32!
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    n_aie_rows = 1
    
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = 512
    
    @device(AIEDevice.npu2)
    def device_body():
        # L2 types
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[dtype_out]]

        # L1 types - m is now 32, so A_l1 is 8x64 (m//4=8)
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # Use SCALAR kernel with DIM_M=32 (processes m=32 rows per call)
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with="mm_32x64x128.o")
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with="mm_32x64x128.o")

        tiles = [[tile(col, row) for col in range(0, n_aie_cols)] for row in range(0, 3)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        A_l3l2_fifos = [None] * n_aie_rows
        A_l2l1_fifos = [None] * n_aie_rows
        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols
        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        for row in range(n_aie_rows):
            A_l3l2_fifos[row] = object_fifo(f"A_L3L2_{row}", shim_tiles[row], mem_tiles[row], 2, A_l2_ty, None, [[(m, k), (mtk // k, m * k), (k, 1)]])
            A_l2l1_fifos[row] = object_fifo(f"A_L2L1_{row}", mem_tiles[row], core_tiles[row][0:n_aie_cols], 2, A_l1_ty,
                [(mtk // k * 4, m * k // 4), (k // k, k), (m // 4, k), (1, 1)],  # flat access, no stride
                [[(k, k), (m // 4, m * k), (1, 1)] for _ in range(n_aie_cols)])  # flat L1 buffer, k*k contiguous
            object_fifo_link(A_l3l2_fifos[row], A_l2l1_fifos[row])

        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(f"B_L3L2_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2_ty)
            B_l2l1_fifos[col] = object_fifo(f"B_L2L1_{col}", mem_tiles[col], [core_tiles[j][col] for j in range(n_aie_rows)], 2, B_l1_ty)
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2_fifos[row][col] = object_fifo(f"C_L1L2_{col}_{row}", core_tiles[row][col], mem_tiles[col], 1, C_l1_ty)
            C_l2l3_fifos[col] = object_fifo(f"C_L2L3_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2_ty)
            of_offsets = [m * n * i for i in range(n_aie_rows)]
            object_fifo_link([C_l1l2_fifos[j][col] for j in range(n_aie_rows)], C_l2l3_fifos[col], of_offsets)

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range((M // m) * (N // n) // (n_aie_cols * n_aie_rows)):
                            elem_out = C_l1l2_fifos[row][col].acquire(ObjectFifoPort.Produce, 1)
                            zero(elem_out)
                            for _ in range_(K // k):
                                elem_in_b = B_l2l1_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                                elem_in_a = A_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul(elem_in_a, elem_in_b, elem_out)
                                A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (n_aie_rows * m, n), (1, 1))
            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            tb_max_n_rows = 4
            input_task_groups = [[] for _ in range(tb_max_n_rows)]
            output_task_groups = [[] for _ in range(tb_max_n_rows)]

            for group_idx in range(num_groups):
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(A_l3l2_fifos[row], A, tap=A_taps[a_base_idx + row], issue_token=False)
                    dma_start_task(a_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(a_task)

                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(B_l3l2_fifos[col], B, tap=B_taps[b_base_idx + col], issue_token=False)
                    dma_start_task(b_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(b_task)

                c_base_idx = group_idx * n_aie_cols
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(C_l2l3_fifos[col], C, tap=C_taps[c_base_idx + col], issue_token=True)
                    dma_start_task(c_task)
                    output_task_groups[group_idx % tb_max_n_rows].append(c_task)

                if (group_idx % tb_max_n_rows == 1) and (group_idx != 1):
                    if output_task_groups[2]: dma_await_task(*output_task_groups[2]); output_task_groups[2] = []
                    if output_task_groups[3]: dma_await_task(*output_task_groups[3]); output_task_groups[3] = []
                    dma_free_task(*input_task_groups[2]); input_task_groups[2] = []
                    dma_free_task(*input_task_groups[3]); input_task_groups[3] = []
                if group_idx % tb_max_n_rows == 3:
                    if output_task_groups[0]: dma_await_task(*output_task_groups[0]); output_task_groups[0] = []
                    if output_task_groups[1]: dma_await_task(*output_task_groups[1]); output_task_groups[1] = []
                    dma_free_task(*input_task_groups[0]); input_task_groups[0] = []
                    dma_free_task(*input_task_groups[1]); input_task_groups[1] = []

            for g in [0, 1, 2, 3]:
                if output_task_groups[g]: dma_await_task(*output_task_groups[g]); dma_free_task(*input_task_groups[g])

main()
