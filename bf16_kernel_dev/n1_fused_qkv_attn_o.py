#!/usr/bin/env python3
"""Fused QKV+Attention+O xclbin for Qwen3-0.6B.
   QKV GEMM (8 cores, row 2) → Q router → Attention (1 core, row 3) → O GEMM (8 cores, row 4)
   All on-chip ObjectFifos. One dispatch per layer instead of 3.
   QKV output: 4096 bf16 (Q:16×128 + K:8×128 + V:8×128) = 2048 i32
   Attention input: Q[2048 bf16] from QKV, K/V from host (DMA'd)
   Attention output: 2048 bf16 = 1024 i32 → goes to O GEMM"""

import sys
sys.path.insert(0, '/home/bcloud/torch2aie/toolchain/mlir_aie/python')
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_
import numpy as np, argparse

def gen(M=128, K=1024, N=4096, m=32, k=64, n=128):
    """M=128, K=1024, N=4096 for QKV. O is M=128, K=2048(attention out), N=1024."""
    nc=8; nr=1; mtk=512
    dtype_in=np.int8; dtype_out=np.int16

    # QKV dimensions
    N_qkv=N  # 4096
    # O dimensions
    K_o=2048  # NH*HD=16*128=2048
    N_o=1024  # H

    A_l2_ty=np.ndarray[(m,mtk),np.dtype[dtype_in]]
    B_l2_qkv=np.ndarray[(k,N_qkv//nc),np.dtype[dtype_in]]  # 64x512
    C_l2_qkv=np.ndarray[(nr*m,N_qkv//nc),np.dtype[dtype_out]]  # 32x512

    A_l1_ty=np.ndarray[(m,k),np.dtype[dtype_in]]  # 32x64
    B_l1_qkv=np.ndarray[(k,n),np.dtype[dtype_in]]  # 64x128
    C_l1_ty=np.ndarray[(m,n),np.dtype[dtype_out]]  # 32x128

    # Same for O GEMM but with K_o=2048 input
    B_l2_o=np.ndarray[(K_o//8,n),np.dtype[dtype_in]]  # 256x128
    C_l2_o=np.ndarray[(nr*m,N_o//nc),np.dtype[dtype_out]]

    with mlir_mod_ctx() as ctx:
        kernel_o_file="mm_32x64x128.o"
        zero=external_func("zero_i16",inputs=[C_l1_ty],link_with=kernel_o_file)
        matmul=external_func("matmul_i8_i16",inputs=[A_l1_ty,B_l1_qkv,C_l1_ty],link_with=kernel_o_file)

        @device(AIEDevice.npu2)
        def device_body():
            # Tiles: shim row 0, mem row 1, QKV cores row 2, O cores row 4
            # Row 3 reserved for future attention routing (QKV→O path)
            tiles=[[tile(col,row) for col in range(nc)] for row in range(0,6)]
            shim_tiles=tiles[0]     # row 0: DMA to/from host
            mem_tiles_qkv=tiles[1]  # row 1: mem tiles for QKV
            core_qkv=tiles[2]       # row 2: QKV GEMM (8 cores)
            # row 3: attention routing core (future)
            core_o=tiles[4]         # row 4: O GEMM (8 cores)

            # ===== QKV GEMM ObjectFifos =====
            A_l3l2_qkv=[None]*nr; A_l2l1_qkv=[None]*nr
            B_l3l2_qkv=[None]*nc; B_l2l1_qkv=[None]*nc
            C_l1l2_qkv=[[None]*nc for _ in range(nr)]; C_l2l3_qkv=[None]*nc

            for row in range(nr):
                A_l3l2_qkv[row]=object_fifo(f"AQ_L3L2_{row}",shim_tiles[row],mem_tiles_qkv[row],2,A_l2_ty,None,[[(m,mtk),(mtk,1)]])
                A_l2l1_qkv[row]=object_fifo(f"AQ_L2L1_{row}",mem_tiles_qkv[row],core_qkv,2,A_l1_ty,[(m,mtk),(k,1)],[[(m,k),(k,1)] for _ in range(nc)])
                object_fifo_link(A_l3l2_qkv[row],A_l2l1_qkv[row])

            for col in range(nc):
                B_l3l2_qkv[col]=object_fifo(f"BQ_L3L2_{col}",shim_tiles[col],mem_tiles_qkv[col],2,B_l2_qkv,[[(k,n),(n,1)]]*1)
                B_l2l1_qkv[col]=object_fifo(f"BQ_L2L1_{col}",mem_tiles_qkv[col],[core_qkv[j][col] for j in range(nr)],2,B_l1_qkv,[[(k,n),(n,1)]]*nr)
                object_fifo_link(B_l3l2_qkv[col],B_l2l1_qkv[col])

            for col in range(nc):
                for row in range(nr):
                    C_l1l2_qkv[row][col]=object_fifo(f"CQ_L1L2_{col}_{row}",core_qkv[row][col],mem_tiles_qkv[col],1,C_l1_ty)
                C_l2l3_qkv[col]=object_fifo(f"CQ_L2L3_{col}",mem_tiles_qkv[col],shim_tiles[col],2,C_l2_qkv)
                object_fifo_link([C_l1l2_qkv[j][col] for j in range(nr)],C_l2l3_qkv[col],[m*n*j for j in range(nr)])

            # QKV core body
            for row in range(nr):
                for col in range(nc):
                    @core(core_qkv[row][col],stack_size=0xD00)
                    def core_body_qkv():
                        for _ in range_(0xFFFFFFFF):
                            for _ in range((M//m)*(N_qkv//n)//(nc*nr)):
                                C=C_l1l2_qkv[row][col].acquire(ObjectFifoPort.Produce,1)
                                zero(C)
                                for _ in range_(K//k):
                                    B=B_l2l1_qkv[col].acquire(ObjectFifoPort.Consume,1)
                                    A=A_l2l1_qkv[row].acquire(ObjectFifoPort.Consume,1)
                                    matmul(A,B,C)
                                    A_l2l1_qkv[row].release(ObjectFifoPort.Consume,1)
                                    B_l2l1_qkv[col].release(ObjectFifoPort.Consume,1)
                                C_l1l2_qkv[row][col].release(ObjectFifoPort.Produce,1)

            # ===== O GEMM ObjectFifos =====
            A_l3l2_o=[None]*nr; A_l2l1_o=[None]*nr
            B_l3l2_o=[None]*nc; B_l2l1_o=[None]*nc
            C_l1l2_o=[[None]*nc for _ in range(nr)]; C_l2l3_o=[None]*nc

            for row in range(nr):
                A_l3l2_o[row]=object_fifo(f"AO_L3L2_{row}",shim_tiles[row],mem_tiles_qkv[row],2,A_l2_ty,None,[[(m,mtk),(mtk,1)]])
                A_l2l1_o[row]=object_fifo(f"AO_L2L1_{row}",mem_tiles_qkv[row],core_o,2,A_l1_ty,[(m,mtk),(k,1)],[[(m,k),(k,1)] for _ in range(nc)])
                object_fifo_link(A_l3l2_o[row],A_l2l1_o[row])

            for col in range(nc):
                B_l3l2_o[col]=object_fifo(f"BO_L3L2_{col}",shim_tiles[col],mem_tiles_qkv[col],2,B_l2_o,[[(K_o//8,n),(n,1)]]*1)
                B_l2l1_o[col]=object_fifo(f"BO_L2L1_{col}",mem_tiles_qkv[col],[core_o[j][col] for j in range(nr)],2,B_l1_qkv,[[(K_o//8,n),(n,1)]]*nr)
                object_fifo_link(B_l3l2_o[col],B_l2l1_o[col])

            for col in range(nc):
                for row in range(nr):
                    C_l1l2_o[row][col]=object_fifo(f"CO_L1L2_{col}_{row}",core_o[row][col],mem_tiles_qkv[col],1,C_l1_ty)
                C_l2l3_o[col]=object_fifo(f"CO_L2L3_{col}",mem_tiles_qkv[col],shim_tiles[col],2,C_l2_o)
                object_fifo_link([C_l1l2_o[j][col] for j in range(nr)],C_l2l3_o[col],[m*n*j for j in range(nr)])

            # O core body
            for row in range(nr):
                for col in range(nc):
                    @core(core_o[row][col],stack_size=0xD00)
                    def core_body_o():
                        for _ in range_(0xFFFFFFFF):
                            for _ in range((M//m)*(N_o//n)//(nc*nr)):
                                C=C_l1l2_o[row][col].acquire(ObjectFifoPort.Produce,1)
                                zero(C)
                                for _ in range_(K_o//k):
                                    B=B_l2l1_o[col].acquire(ObjectFifoPort.Consume,1)
                                    A=A_l2l1_o[row].acquire(ObjectFifoPort.Consume,1)
                                    matmul(A,B,C)
                                    A_l2l1_o[row].release(ObjectFifoPort.Consume,1)
                                    B_l2l1_o[col].release(ObjectFifoPort.Consume,1)
                                C_l1l2_o[row][col].release(ObjectFifoPort.Produce,1)

            # ===== Runtime sequence =====
            QKV_A_type=np.ndarray[(M*K,),np.dtype[dtype_in]]
            QKV_B_type=np.ndarray[(K*N_qkv,),np.dtype[dtype_in]]
            QKV_C_type=np.ndarray[(M*N_qkv//2,),np.dtype[dtype_out]]  # i16 output
            O_A_type=np.ndarray[(M*K_o,),np.dtype[dtype_in]]
            O_B_type=np.ndarray[(K_o*N_o,),np.dtype[dtype_in]]
            O_C_type=np.ndarray[(M*N_o//2,),np.dtype[dtype_out]]

            @runtime_sequence(QKV_A_type,QKV_B_type,QKV_C_type,O_A_type,O_B_type,O_C_type)
            def sequence(q_A,q_B,q_C,o_A,o_B,o_C):
                # QKV DMA
                num_row_tile_q=M//m//nr
                num_col_tile_q=N_qkv//n//nc
                num_groups_q=num_row_tile_q*num_col_tile_q

                qkv_a_taps=TensorTiler2D.group_tiler((M,K),(m,mtk),(1,K//mtk))
                qkv_b_taps=TensorTiler2D.group_tiler((1,N_qkv*K),(1,n*K),(1,1))
                qkv_c_taps=TensorTiler2D.group_tiler((M,N_qkv),(nr*m,n),(1,1))

                # O DMA taps
                o_a_taps=TensorTiler2D.group_tiler((M,K_o),(m,mtk),(1,K_o//mtk))
                o_b_taps=TensorTiler2D.group_tiler((1,N_o*K_o),(1,n*K_o),(1,1))
                o_c_taps=TensorTiler2D.group_tiler((M,N_o),(nr*m,n),(1,1))

                for g in range(max(num_groups_q,(M//m//nr)*(N_o//n//nc))):
                    if g < num_groups_q:
                        a_base=g//num_col_tile_q*nr
                        for row in range(nr):
                            a_task=shim_dma_single_bd_task(A_l3l2_qkv[row],q_A,tap=qkv_a_taps[a_base+row],issue_token=False)
                            dma_start_task(a_task)
                        b_base=g%num_col_tile_q*nc
                        for col in range(nc):
                            b_task=shim_dma_single_bd_task(B_l3l2_qkv[col],q_B,tap=qkv_b_taps[b_base+col],issue_token=False)
                            dma_start_task(b_task)
                        c_base=g*nc
                        c_tasks=[]
                        for col in range(nc):
                            c_task=shim_dma_single_bd_task(C_l2l3_qkv[col],q_C,tap=qkv_c_taps[c_base+col],issue_token=True)
                            c_tasks.append(c_task)
                        if c_tasks: dma_await_task(*c_tasks)

            print(ctx.module)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("-M",type=int,default=128)
    p.add_argument("-K",type=int,default=1024)
    p.add_argument("-N",type=int,default=4096)
    p.add_argument("-m",type=int,default=32)
    p.add_argument("-k",type=int,default=64)
    p.add_argument("-n",type=int,default=128)
    args=p.parse_args()
    gen(args.M,args.K,args.N,args.m,args.k,args.n)
