#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

static float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, 4); return f;
}
static uint16_t float_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

int main() {
    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string("/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_128x1024x1024_128x64x128.xclbin"));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");
    FILE* f = fopen("/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_128x1024x1024_128x64x128.txt","rb");
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(),4,instr.size(),f); fclose(f);
    
    int M=128, K=1024, N=1024;
    int a_sz=M*K*2, b_sz=K*N*9/8, c_sz=M*N*2;
    
    auto bo_instr=xrt::bo(device,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,kernel.group_id(1));
    auto bo_a=xrt::bo(device,a_sz,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(3));
    auto bo_b=xrt::bo(device,b_sz,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(4));
    auto bo_c=xrt::bo(device,c_sz,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(5));
    
    memcpy(bo_instr.map(),instr.data(),instr.size()*4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE,instr.size()*4,0);
    
    // Test A as bf16, all 1.0
    auto* a=(uint16_t*)bo_a.map();
    for(int i=0;i<M*K;i++) a[i]=float_to_bf16(1.0f);
    
    // B as bf16 packed format = floatToBfp16(8, K*N, bf16_values, 0)
    // For all 1.0 bf16: each 8-element group shares exponent and 8-bit mantissa
    // The packing is: [exp_u8][m0_s8][m1_s8]...[m7_s8] for each group
    // For all-1.0 bf16 (0x3F80): exp=127, mantissa=0x80=128 as u8
    // Group exp = 127, mantissas = all 128 (0x80)
    auto* b=(uint8_t*)bo_b.map();
    int ngroups=K*N/8;
    for(int i=0;i<ngroups;i++){
        b[i*9]=127;  // shared exponent
        for(int j=0;j<8;j++) b[i*9+1+j]=128;  // mantissa
    }
    
    auto* c=(uint16_t*)bo_c.map();
    memset(c,0,c_sz);
    
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE,a_sz,0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE,b_sz,0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE,c_sz,0);
    
    auto run=kernel((unsigned)3,bo_instr,(unsigned)instr.size(),bo_a,bo_b,bo_c);
    run.wait(); bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE,c_sz,0);
    
    printf("BFP16 test (all 1.0):\n");
    printf("  C[0]=%f (expect %f)\n", bf16_to_float(c[0]), 1024.0f);
    printf("  C[1]=%f (expect %f)\n", bf16_to_float(c[1]), 1024.0f);
    
    // Check all rows
    int correct=0;
    for(int i=0;i<std::min(M*N,10000);i++){
        if(fabsf(bf16_to_float(c[i])-1024.0f)<1.0f) correct++;
    }
    printf("  Correct: %d/10000\n", correct);
    
    // Test with random values in A
    srand(42);
    for(int i=0;i<M*K;i++) a[i]=float_to_bf16((float)(rand()%10-5));
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE,a_sz,0);
    run=kernel((unsigned)3,bo_instr,(unsigned)instr.size(),bo_a,bo_b,bo_c);
    run.wait(); bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE,c_sz,0);
    
    // Reference: sum of A values * 1.0 = sum of A per row
    printf("\nBFP16 random test:\n");
    for(int m=0;m<5;m++){
        float sum=0;
        for(int k=0;k<K;k++) sum+=bf16_to_float(a[m*K+k]);
        printf("  Row %d: NPU=%f REF=%f\n", m, bf16_to_float(c[m*N]), sum);
    }
}
