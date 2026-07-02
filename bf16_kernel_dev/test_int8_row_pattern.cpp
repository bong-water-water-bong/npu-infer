// test_int8_row_pattern.cpp - Row-pattern test for INT8 xclbin
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

int main() {
    const char* xclbin = "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV.xclbin";
    const char* insts_p = "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV.txt";
    int M=128, K=1024, N=1024;
    
    FILE* f = fopen(insts_p, "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(), 4, instr.size(), f); fclose(f);
    
    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string(xclbin));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");
    
    auto bo_instr = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    
    memcpy(bo_instr.map(), instr.data(), instr.size()*4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE, instr.size()*4, 0);
    
    auto* a = (int8_t*)bo_a.map();
    auto* b = (int8_t*)bo_b.map();
    auto* c = (int16_t*)bo_c.map();
    
    // A: each row has row_number (0..127)
    for (int i = 0; i < M*K; i++) a[i] = i / K;
    
    // B: all 1s
    memset(b, 1, K*N);
    
    // C: zero
    memset(c, 0, M*N*2);
    
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
    
    printf("Expected: C[m][n] = m * K = m * %d\n", K);
    printf("A[0]=%d, A[%d]=%d\n", a[0], (M-1)*K, a[(M-1)*K]);
    
    auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
    
    printf("\nNPU results:\n");
    printf("  C[0]   = %d (expected %d)\n", c[0], K*0);
    printf("  C[%d]  = %d (expected %d)\n", N, c[N], K*1);
    printf("  C[%d] = %d (expected %d)\n", 2*N, c[2*N], K*2);
    printf("  C[%d] = %d (expected %d)\n", 3*N, c[3*N], K*3);
    printf("  C[%d]= %d (expected %d)\n", 31*N, c[31*N], K*31);
    printf("  C[%d]= %d (expected %d)\n", 32*N, c[32*N], K*32);
    printf("  C[%d]= %d (expected %d)\n", 63*N, c[63*N], K*63);
    printf("  C[%d]= %d (expected %d)\n", 64*N, c[64*N], K*64);
    printf("  C[%d]= %d (expected %d)\n", 95*N, c[95*N], K*95);
    printf("  C[%d]= %d (expected %d)\n", 96*N, c[96*N], K*96);
    printf("  C[%d]= %d (expected %d)\n", 127*N, c[127*N], K*127);
    
    int correct = 0, wrong = 0;
    for (int m = 0; m < M; m++) {
        if (c[m*N] == K*m) correct++; else wrong++;
    }
    printf("\nRows correct: %d, wrong: %d\n", correct, wrong);
    
    return wrong > 0 ? 1 : 0;
}
