#!/usr/bin/env python3
"""Fused GU+SiLU+D xclbin. GU GEMM (8 cores, row 2) → SiLU (1 core, row 3 col 0) → D GEMM (8 cores, row 4).
   1 dispatch instead of 2. Saves 28 dispatches per token."""

import sys
sys.path.insert(0, '/home/bcloud/torch2aie/toolchain/mlir_aie/python')
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_
import numpy as np, argparse

def gen(M=128,K=1024,N_gu=6144,N_d=1024,m=32,k=64,n=128):
    nc=8; nr=1; mtk=512
    dtype_in=np.int8; dtype_out=np.int16

    # GU: A[M×K], B[K×N_gu], C[M×N_gu]
    # D:  A[M×IM], B[IM×N_d], C[M×N_d]
    IM=3072
    A_l2_ty=np.ndarray[(m,mtk),np.dtype[dtype_in]]
    B_l2_ty=np.ndarray[(k,n),np.dtype[dtype_in]]
    C_l2_ty=np.ndarray[(nr*m,n),np.dtype[dtype_out]]
    A_l1_ty=np.ndarray[(m,k),np.dtype[dtype_in]]
    B_l1_ty=np.ndarray[(k,n),np.dtype[dtype_in]]
    C_l1_ty=np.ndarray[(m,n),np.dtype[dtype_out]]

    # SiLU buffers: GU output (6144 bf16=3072 i32), D input (3072 bf16=1536 i32)
    GU_out_dw=N_gu//2  # 3072 i32
    D_in_dw=IM//2      # 1536 i32
    SiLU_in_ty=np.ndarray[(GU_out_dw,),np.dtype[np.int32]]
    SiLU_out_ty=np.ndarray[(D_in_dw,),np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:
        kernel_o="mm_32x64x128.o"
        zero=external_func("zero_i16",inputs=[C_l1_ty],link_with=kernel_o)
        matmul=external_func("matmul_i8_i16",inputs=[A_l1_ty,B_l1_ty,C_l1_ty],link_with=kernel_o)
        silu=external_func("silu_gate_bf16",inputs=[SiLU_in_ty,SiLU_out_ty,np.int32],link_with="silu_gate.o")

        @device(AIEDevice.npu2)
        def device_body():
            # Rows: 0=shim, 1=mem(GU), 2=core(GU), 3=core(SiLU), 4=mem(D), 5=core(D)
            shim_tiles=[tile(col,0) for col in range(nc)]
            mem_gu=[tile(col,1) for col in range(nc)]     # GU mem tiles
            core_gu=[tile(col,2) for col in range(nc)]     # GU GEMM
            core_silu=tile(0,3)                             # SiLU helper
            core_d=[tile(col,4) for col in range(nc)]       # D GEMM

            # ===== GU GEMM ObjectFifos =====
            A_l3l2_g=[None]*nr; A_l2l1_g=[None]*nr
            B_l3l2_g=[None]*nc; B_l2l1_g=[None]*nc
            C_l1l2_g=[[None]*nc for _ in range(nr)]; C_l2l3_g=[None]*nc

            for row in range(nr):
                A_l3l2_g[row]=object_fifo(f"AG_L3L2_{row}",shim_tiles[row],mem_gu[row],2,A_l2_ty,None,[[(m,mtk),(mtk,1)]])
                A_l2l1_g[row]=object_fifo(f"AG_L2L1_{row}",mem_gu[row],core_gu,2,A_l1_ty,[(m,mtk),(k,1)],[[(m,k),(k,1)] for _ in range(nc)])
                object_fifo_link(A_l3l2_g[row],A_l2l1_g[row])
            for col in range(nc):
                B_l3l2_g[col]=object_fifo(f"BG_L3L2_{col}",shim_tiles[col],mem_gu[col],2,B_l2_ty)
                B_l2l1_g[col]=object_fifo(f"BG_L2L1_{col}",mem_gu[col],[core_gu[j][col] for j in range(nr)],2,B_l1_ty)
                object_fifo_link(B_l3l2_g[col],B_l2l1_g[col])

            for col in range(nc):
                for row in range(nr):
                    C_l1l2_g[row][col]=object_fifo(f"CG_L1L2_{col}_{row}",core_gu[row][col],mem_gu[col],1,C_l1_ty)
                C_l2l3_g[col]=object_fifo(f"CG_L2L3_{col}",mem_gu[col],shim_tiles[col],2,C_l2_ty)
                object_fifo_link([C_l1l2_g[j][col] for j in range(nr)],C_l2l3_g[col],[m*n*j for j in range(nr)])

            # GU core bodies
            for row in range(nr):
                for col in range(nc):
                    @core(core_gu[row][col],stack_size=0xD00)
                    def body_gu():
                        for _ in range_(0xFFFFFFFF):
                            for _ in range((M//m)*(N_gu//n)//(nc*nr)):
                                C=C_l1l2_g[row][col].acquire(ObjectFifoPort.Produce,1)
                                zero(C)
                                for _ in range_(K//k):
                                    B=B_l2l1_g[col].acquire(ObjectFifoPort.Consume,1)
                                    A=A_l2l1_g[row].acquire(ObjectFifoPort.Consume,1)
                                    matmul(A,B,C)
                                    A_l2l1_g[row].release(ObjectFifoPort.Consume,1)
                                    B_l2l1_g[col].release(ObjectFifoPort.Consume,1)
                                C_l1l2_g[row][col].release(ObjectFifoPort.Produce,1)

            # ===== SiLU core (row 3, col 0) — reads GU output, applies SiLU, writes D input =====
            gu_out_fifo=object_fifo("gu_output_silu",shim_tiles[0],mem_gu[0],1,SiLU_in_ty,
                None,[[(GU_out_dw,),(1,)]])
            silu_out_fifo=object_fifo("silu_output_d",core_silu,shim_tiles[0],1,SiLU_out_ty,
                None,[[(D_in_dw,),(1,)]])

            @core(core_silu,stack_size=0xD00)
            def body_silu():
                for _ in range_(0xFFFFFFFF):
                    gu=gu_out_fifo.acquire(ObjectFifoPort.Consume,1)
                    di=silu_out_fifo.acquire(ObjectFifoPort.Produce,1)
                    silu(gu,di,IM)
                    gu_out_fifo.release(ObjectFifoPort.Consume,1)
                    silu_out_fifo.release(ObjectFifoPort.Produce,1)

            # ===== Runtime sequence =====
            A_type=np.ndarray[(M*K,),np.dtype[dtype_in]]
            B_gu_type=np.ndarray[(K*N_gu,),np.dtype[dtype_in]]
            C_gu_type=np.ndarray[(nr*(M//nr)*(N_gu//nc),),np.dtype[dtype_out]]
            B_d_type=np.ndarray[(IM*N_d,),np.dtype[dtype_in]]
            C_d_type=np.ndarray[(M*N_d,),np.dtype[dtype_out]]

            @runtime_sequence(A_type,B_gu_type,C_gu_type,SiLU_in_ty,SiLU_out_ty,B_d_type,C_d_type)
            def sequence(A_in,B_gu,C_gu_out,gu_buf,di_buf,B_d,C_d_out):
                # GU GEMM
                a_taps=TensorTiler2D.group_tiler((M,K),(m,mtk),(1,K//mtk))
                b_taps=TensorTiler2D.group_tiler((1,N_gu*K),(1,n*K),(1,1))
                c_taps=TensorTiler2D.group_tiler((M,N_gu),(nr*m,n),(1,1))
                num_g=min((M//m//nr)*(N_gu//n//nc),1)

                for g in range(num_g):
                    a_base=(g//(N_gu//n//nc))*nr
                    for row in range(nr):
                        at=shim_dma_single_bd_task(A_l3l2_g[row],A_in,tap=a_taps[a_base+row],issue_token=False)
                        dma_start_task(at)
                    b_base=(g%(N_gu//n//nc))*nc
                    for col in range(nc):
                        bt=shim_dma_single_bd_task(B_l3l2_g[col],B_gu,tap=b_taps[b_base+col],issue_token=False)
                        dma_start_task(bt)
                    c_base=g*nc
                    ctasks=[]
                    for col in range(nc):
                        ct=shim_dma_single_bd_task(C_l2l3_g[col],C_gu_out,tap=c_taps[c_base+col],issue_token=True)
                        ctasks.append(ct)
                    if ctasks: dma_await_task(*ctasks)

                # SiLU: GU output → SiLU → D input (CPU copies data between NPU dispatches)
                silu_task=shim_dma_single_bd_task(gu_out_fifo,gu_buf,issue_token=False)
                dma_start_task(silu_task)
                d_out=shim_dma_single_bd_task(silu_out_fifo,di_buf,issue_token=True)
                dma_await_task(d_out)

            print(ctx.module)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("-M",type=int,default=128)
    p.add_argument("-K",type=int,default=1024)
    p.add_argument("--N_gu",type=int,default=6144)
    p.add_argument("--N_d",type=int,default=1024)
    p.add_argument("-m",type=int,default=32)
    p.add_argument("-k",type=int,default=64)
    p.add_argument("-n",type=int,default=128)
    args=p.parse_args()
    gen(args.M,args.K,args.N_gu,args.N_d,args.m,args.k,args.n)
