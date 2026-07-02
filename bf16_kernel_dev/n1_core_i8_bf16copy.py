#!/usr/bin/env python3
# Exact copy of bf16 design (config1/n1_core_bf16.py) but with int8/i16 types
import argparse
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    n_aie_cols = 8
    dtype_in = np.int8
    dtype_out = np.int16

    r = 1
    s = 1
    t = 1
    mtk = 512

    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        kernel_o = "mm_128x64x128.o"
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)

        tiles = [
            [tile(col, row) for col in range(n_aie_cols)] for row in range(3)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        A_l3l2_fifos = [None]
        A_l2l1_fifos = [None]

        A_l3l2_fifos[0] = object_fifo(
            f"A_L3L2_{0}",
            shim_tiles[0],
            mem_tiles[0],
            2,
            A_l2_ty,
            None,
            [[(m, k), (mtk // k, m * k), (k, 1)]],
        )
        A_l2l1_fifos[0] = object_fifo(
            f"A_L2L1_{0}",
            mem_tiles[0],
            core_tiles[0][0:n_aie_cols],
            2,
            A_l1_ty,
            [(mtk // k * 4, m * k // 4), (k // s, s), (m // 4, k), (s, 1)],
            [[(k // s, r * s), (m // 4 // r, r * k), (r * s, 1)] for _ in range(n_aie_cols)],
        )
        object_fifo_link(A_l3l2_fifos[0], A_l2l1_fifos[0])

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols
        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(1)]
        C_l2l3_fifos = [None] * n_aie_cols

        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                2,
                B_l2_ty,
            )
            B_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [core_tiles[j][col] for j in range(1)],
                2,
                B_l1_ty,
            )
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

        for col in range(n_aie_cols):
            C_l1l2_fifos[0][col] = object_fifo(
                f"C_L1L2_{col}_{0}",
                core_tiles[0][col],
                mem_tiles[col],
                1,
                C_l1_ty,
            )
            C_transformations = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
            C_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                2,
                C_l2_ty,
                C_transformations,
            )
            of_offsets = [m * n * i for i in range(1)]
            object_fifo_link(
                [C_l1l2_fifos[j][col] for j in range(1)],
                C_l2l3_fifos[col],
                of_offsets,
            )

        for col in range(n_aie_cols):
            @core(core_tiles[0][col], stack_size=0x400)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range(
                        (M // m) * (N // n) // (n_aie_cols)
                    ):
                        elem_out = C_l1l2_fifos[0][col].acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        zero(elem_out)
                        for _ in range_(K // k):
                            elem_in_b = B_l2l1_fifos[col].acquire(
                                ObjectFifoPort.Consume, 1
                            )
                            for _ in range(4):
                                elem_in_a = A_l2l1_fifos[0].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                matmul(elem_in_a, elem_in_b, elem_out)
                                A_l2l1_fifos[0].release(ObjectFifoPort.Consume, 1)
                            B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                        C_l1l2_fifos[0][col].release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
            B_taps = TensorTiler2D.group_tiler((1, N * K), (1, n * K), (1, 1))
            C_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, 1))

            num_row_tile = M // m
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            tb_max_n_rows = 4
            input_task_groups = [[] for _ in range(tb_max_n_rows)]
            output_task_groups = [[] for _ in range(tb_max_n_rows)]

            for group_idx in range(num_groups):
                a_base_idx = group_idx // num_col_tile
                a_task = shim_dma_single_bd_task(
                    A_l3l2_fifos[0], A,
                    tap=A_taps[a_base_idx],
                    issue_token=False,
                )
                dma_start_task(a_task)
                input_task_groups[group_idx % tb_max_n_rows].append(a_task)

                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(
                        B_l3l2_fifos[col], B,
                        tap=B_taps[b_base_idx + col],
                        issue_token=False,
                    )
                    dma_start_task(b_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(b_task)

                c_base_idx = group_idx * n_aie_cols
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(
                        C_l2l3_fifos[col], C,
                        tap=C_taps[c_base_idx + col],
                        issue_token=True,
                    )
                    dma_start_task(c_task)
                    output_task_groups[group_idx % tb_max_n_rows].append(c_task)

                if (group_idx % tb_max_n_rows == 1) and (group_idx != 1):
                    if output_task_groups[2]:
                        dma_await_task(*output_task_groups[2]); output_task_groups[2] = []
                        dma_free_task(*input_task_groups[2]); input_task_groups[2] = []
                    if output_task_groups[3]:
                        dma_await_task(*output_task_groups[3]); output_task_groups[3] = []
                        dma_free_task(*input_task_groups[3]); input_task_groups[3] = []
                if group_idx % tb_max_n_rows == 3:
                    if output_task_groups[0]:
                        dma_await_task(*output_task_groups[0]); output_task_groups[0] = []
                        dma_free_task(*input_task_groups[0]); input_task_groups[0] = []
                    if output_task_groups[1]:
                        dma_await_task(*output_task_groups[1]); output_task_groups[1] = []
                        dma_free_task(*input_task_groups[1]); input_task_groups[1] = []

            for g in [0, 1, 2, 3]:
                if output_task_groups[g]:
                    dma_await_task(*output_task_groups[g])
                    dma_free_task(*input_task_groups[g])

main()
