#!/usr/bin/env python3
"""
Generate MLIR for BF16-native GEMM with 64×64 B tiles, depth=2.
Fits in 56KB SRAM: A=8KB, B=16KB (depth 2), C=32KB.

Usage: python gen_mlir_v2.py > design_bf16_v2.mlir
"""
import os, sys
from pathlib import Path

# Use the 40-column toolchain's Python MLIR-AIE bindings
TC40 = Path("/home/bcloud/mlir-aie/npu2_40_toolchain")
os.environ.setdefault("TORCH2AIE_TOOLCHAIN", str(TC40))
os.environ.setdefault("AIETOOLS_DIR", "/home/bcloud/torch2aie/toolchain/aietools")
os.environ.setdefault("MLIR_AIE_DIR", str(TC40))

# Add toolchain Python paths for MLIR-AIE dialect
tc_python = str(TC40 / "python")
tc_libs = str(TC40 / "lib")
sys.path.insert(0, tc_python)
sys.path.insert(0, "/home/bcloud/torch2aie/examples")

os.environ.setdefault("LD_LIBRARY_PATH", f"{tc_libs}:{tc_python}/_mlir_libs")
os.environ.setdefault("MLIR_RUNNER_UTILS_DIR", tc_libs)

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from ml_dtypes import bfloat16
import numpy as np

def gen_design(M, K, N, m, k, n, cols=8, rows=4):
    """Generate BF16 MLIR with 64×64 B tiles, depth=2."""
    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu2)
        def body():
            # Types
            T_i = bfloat16  # input
            T_o = bfloat16  # output
            
            # L1 tile types (compute tile level)
            A_l1 = np.ndarray[(m // 4, k), np.dtype[T_i]]   # 32×64
            B_l1 = np.ndarray[(k, n), np.dtype[T_i]]         # 64×64
            C_l1 = np.ndarray[(m, n), np.dtype[T_o]]         # 128×64
            
            # L2 tile types (mem tile level)
            A_l2 = np.ndarray[(m, 512), np.dtype[T_i]]       # 128×512
            B_l2 = np.ndarray[(k, n), np.dtype[T_i]]         # 64×64  (same at L2)
            C_l2 = np.ndarray[(rows * m, n), np.dtype[T_o]]  # 512×64
            
            # External kernel functions
            zero = external_func(
                "zero_kernel_bf16_v2",
                inputs=[C_l1],
                link_with="mm_bf16_v2.o",
            )
            matmul = external_func(
                "matmul_bf16_v2",
                inputs=[A_l1, B_l1, C_l1],
                link_with="mm_bf16_v2.o",
            )
            
            # Create tile grid
            tiles = [[tile(col, row) for col in range(cols)] for row in range(2 + rows)]
            shim_tiles = tiles[0]
            mem_tiles = tiles[1]
            core_tiles = tiles[2:]
            
            # A ObjectFIFOs (activation): shim → mem → core (column broadcast)
            for row in range(rows):
                shim_col = (row * 2) % cols
                a_l3 = object_fifo(
                    f"A_L3_{row}", shim_tiles[shim_col], mem_tiles[shim_col], 2, A_l2,
                    None,
                    [[(m, k), (512 // k, m * k), (k, 1)]],
                )
                a_l2 = object_fifo(
                    f"A_L2_{row}", mem_tiles[shim_col],
                    [core_tiles[row][c] for c in range(cols)],
                    2, A_l1,
                    [(512 // k * 4, m * k // 4), (k // 8, 8), (m // 4, k), (8, 1)],
                    [[(k // 8, 2 * 8), (m // 4 // 2, 2 * k), (2 * 8, 1)]
                     for _ in range(cols)],
                )
                object_fifo_link(a_l3, a_l2)
            
            # B ObjectFIFOs (weights): shim → mem → core (row broadcast)
            for col in range(cols):
                b_l3 = object_fifo(
                    f"B_L3_{col}", shim_tiles[col], mem_tiles[col], 2, B_l2,
                )
                b_l2 = object_fifo(
                    f"B_L2_{col}", mem_tiles[col],
                    [core_tiles[row][col] for row in range(rows)],
                    2, B_l1,  # depth=2 with 8KB tiles = 16KB total
                )
                object_fifo_link(b_l3, b_l2)
            
            # C ObjectFIFOs (output): core → mem → shim
            for col in range(cols):
                for row in range(rows):
                    c_l1 = object_fifo(
                        f"C_L1_{col}_{row}", core_tiles[row][col], mem_tiles[col], 1, C_l1,
                    )
                c_l2 = object_fifo(
                    f"C_L2_{col}", mem_tiles[col], shim_tiles[col], 2, C_l2,
                    [(m // 2, 2 * n), (2, 8), (n // 8, 2 * 8), (8, 1)],
                )
                of_offsets = [m * n * i for i in range(rows)]
                object_fifo_link(
                    [C_l1 for _ in range(rows)], c_l2,
                    of_offsets,
                )
            
            # Core functions: endless loop over K-tiles
            for col in range(cols):
                for row in range(rows):
                    core_tile = core_tiles[row][col]
                    a_fifo = f"A_L2_{row}"
                    b_fifo = f"B_L2_{col}"
                    c_fifo = f"C_L1_{col}_{row}"
                    
                    @core(core_tile, f"core_{col}_{row}")
                    def core_fn():
                        n_k_tiles = K // k
                        n_c_phases = (M // m)
                        # Endless loop for streaming
                        for _ in range_(0xFFFFFFFF):
                            for _ in range_(n_c_phases):
                                # Zero C tile
                                c_acc = acquire(c_fifo, Produce, 1)
                                zero(c_acc)
                                
                                # Accumulate over K
                                for _ in range_(n_k_tiles):
                                    b_buf = acquire(b_fifo, Consume, 1)
                                    for _ in range_(4):  # 4 × 32-row A tiles = 128 rows
                                        a_buf = acquire(a_fifo, Consume, 1)
                                        matmul(a_buf, b_buf, c_acc)
                                        release(a_fifo, Consume, 1)
                                    release(b_fifo, Consume, 1)
                                
                                release(c_fifo, Produce, 1)
        
        print(ctx.module)


if __name__ == "__main__":
    M = 1024
    K = 2048
    N = 2048
    m = 128   # tile M
    k = 64    # tile K  
    n = 64    # tile N (reduced from 128→64 to fit BF16 B tiles with depth=2)
    gen_design(M, K, N, m, k, n)
