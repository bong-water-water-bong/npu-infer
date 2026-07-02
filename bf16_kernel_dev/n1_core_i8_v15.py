#!/usr/bin/env python3
#
# INT8 MLIR generator v15 — TWO COLUMN, ONE CORE
# - shim[0], mem[0,1], core[0,2]: A fifo chain
# - shim[1], mem[1,1], core[1,2]: B fifo chain  
# - C output from core[0,2] → mem[0,1] → shim[0] (or use core[1,2] for C)
# - Single core at (0,2) processes matmul
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
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-m", type=int, default=128)
    parser.add_argument("-k", type=int, default=64)
    parser.add_argument("-n", type=int, default=128)
    args = parser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)

def my_matmul(M, K, N, m, k, n):
    dtype_in = np.int8
    dtype_out = np.int16
    mtk = 512
    
    @device(AIEDevice.npu2)
    def device_body():
        A_l2_ty = np.ndarray[(m, mtk), np.dtype[dtype_in]]
        A_l1_ty = np.ndarray[(m // 4, k), np.dtype[dtype_in]]  # 32x64 for m=128
        B_l2_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        C_l2_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
        
        kernel_o = "mm_128x64x128_DIM128.o"  # DIM_M=128
        zero = external_func("zero_scalar_i16", inputs=[C_l1_ty], link_with=kernel_o)
        matmul = external_func("matmul_scalar_i8_i16", inputs=[A_l1_ty, B_l1_ty, C_l1_ty], link_with=kernel_o)
        
        sA = tile(0, 0)
        mA = tile(0, 1)
        cA = tile(0, 2)
        
        sB = tile(1, 0)
        mB = tile(1, 1)
        cB = tile(1, 2)
        
        # A on col 0: shim[0] → mem[0,1] → core[0,2]
        A_L3L2 = object_fifo("A_L3L2", sA, mA, 2, A_l2_ty)
        A_L2L1 = object_fifo("A_L2L1", mA, cA, 2, A_l1_ty)
        object_fifo_link(A_L3L2, A_L2L1)
        
        # B on col 1: shim[1] → mem[1,1] → core[1,2]
        B_L3L2 = object_fifo("B_L3L2", sB, mB, 2, B_l2_ty)
        B_L2L1 = object_fifo("B_L2L1", mB, cB, 2, B_l1_ty)
        object_fifo_link(B_L3L2, B_L2L1)
        
        # We need ONE core. Both core[0,2] and core[1,2] get data from independent paths.
        # But matmul needs both A and B in the SAME core.
        # Solution: route A through mem[0,1] to core[0,2], route B through mem[1,1] to core[0,2]
        # via cross-column routing. But that requires mem[0,1] South → mem[1,1] North connections.
        
        # Alternative: use core[1,2] and route A from mem[0,1] to core[1,2].
        # The route is: shim[0]→mem[0,1]→switchbox→core[1,2] via horizontal connection.
        # This is complex. Let me use a simpler approach: 
        # A and B on different columns, C on one column.
        
        # Actually the simplest approach for 1 core: share the same column.
        # Put A and B fifos through the same mem tile but with separate DMA channels.
        # The NPU has multiple DMA channels per tile.
        
        # Single column approach, dual DMA:
        # A: DMA channel 0 on shim/mem/core
        # B: DMA channel 1 on shim/mem/core
        
        # But we showed this fails (v13). Let me try routing B through a 
        # different path: shim[1] → mem[0,1] via horizontal → core[0,2]
        
        # Actually, let me just use A and B on the SAME column but with
        # explicit buffer routing. v13 failed but let me look at why.
        
        # The issue was "NPU lowering pipeline failed" — routing couldn't put
        # A and B through the same mem tile. Let me try with explicit DMA channels.
        
        # Actually, looking at the BFP16: A goes through mem[0,1] and B through 
        # mem[0,1] too (col 0). The difference: BFP16 uses different columns
        # for B: B goes through mem[col,1] for col=0..7.
        
        # For single core: we just need col 0. Let me try without linking:
        # A_L3L2 = separate fifo
        # A_L2L1 = separate fifo  
        # B_L3L2 and B_L2L1 separate too (all on col 0)
        
        # ... but v13 already tried this. Let me try WITHOUT object_fifo_link.
        # Use separate fifos with different depths.
        
        # SIMPLEST: Just use the BFP16 design but with INT8 types.
        # That means: 1 shared A fifo, 8 B fifos, 8 cores.
        # The K-slice interleaving still happens.
        
        # DIFFERENT APPROACH: instead of per-core fifos, use software reordering.
        # Feed data such that the interleaving produces correct results.
        # Each core computes C += Σ A(K_interleaved) × B(K_correct).
        # If A(K_interleaved) = A(K_correct) (which is true if A is 
        # K-invariant in each block of 8), then result is correct.
        
        # For inference: A = activations (random per K), B = weights (fixed).
        # We can REORDER THE WEIGHTS B to match A's K-interleaving!
        # 
        # Specifically, reorder B such that B[k][n] for interleaved k positions 
        # matches what A actually provides. Each core gets A(K[i*64:(i+1)*64]) 
        # for ONE specific i (its assigned sub-view) and repeats it for all 
        # 16 K-iterations. If we put ALL K-values of B at that A's k-range 
        # into the B fifo, then the matmul computes:
        # Σ_{k'=0}^{63} A[m][k'] × B[k'][n] — only using 64 K-values.
        
        # This gives Σ_{k in A_subview} A[m][k] × B[k][n], which is the 
        # correct partial sum for the FULL K-range if B is reordered so
        # that for each k in A_subview (64 values), B[k][n] = Σ_{correct_k} B_correct[correct_k][n].
        
        # That's NOT how matmul works. Each output element uses ONE k at a time.
        
        # THE CORRECT FIX: each core must process ALL K-values exactly once.
        # With shared A fifo, each core's 16 acquires across K-loop get
        # different sub-views. For this, the sub-view must ADVANCE per 
        # core per iteration, which requires the fifo to have MORE sub-views
        # than cores (N_subviews > N_cores).
        
        # With mtk=512, m=32, k=64: 8 sub-views. N_cores=8, N_subviews=8.
        # Each core gets sub-view fixed_idx.
        # 
        # With mtk=1024, m=32, k=64: 16 sub-views. N_cores=8.
        # Core 0: sub-views 0,8 (cycle once per 2 K-loop iterations)
        # Core 1: sub-views 1,9
        # ...
        # Core 7: sub-views 7,15
        # 
        # For a K-loop of 16: each core does 16 acquires. 16×8=128 fifo accesses.
        # 128 accesses / 16 sub-views = 8 cycles re-acquiring the same sub-views.
        
        # This STILL doesn't give every core every sub-view!
        # EACH CORE MUST ACQUIRE FROM ALL SUB-VIEWS.
        # With shared fifo and N_cores=8: impossible due to round-robin.
        
        # THE ONLY SOLUTION IS PER-CORE FIFOS.
        # And mem tile memory limit prevents this at standard sizes.
        
        # But what if each core has its OWN A fifo that's SMALLER?
        # mtk=64 (same as k): A_l2 = memref<32x64xi8> = 2KB.
        # A_l1 can be the SAME buffer (no K-split).
        # Each core's A fifo reads 32×64 = 2KB from L3.
        # Depth 2 = 4KB per core.
        
        # With mtk=k: each A element covers exactly ONE K-tile.
        # K-loop: 16 iterations, 16 A elements needed.
        # BD reads 16 × 2KB = 32KB per compute group.
        # 8 cores × 4KB = 32KB total A memory in all mem tiles.
        # Each mem tile: A=4KB + B=16KB + C=8KB = 28KB < 64KB!
        
        # THIS SHOULD WORK! Let me implement.

        pass

main()
