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
    const char* xclbin = "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV_v2.xclbin";
    const char* insts_p = "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV_v2.txt";
    int M=128, K=1024, N=1024;
    
    FILE* f = fopen(insts_p, "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
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
    
    // A: row value = row index
    for (int i = 0; i < M*K; i++) a[i] = i / K;
    // B: all 1s
    memset(b, 1, K*N);
    // C: zero
    memset(c, 0, M*N*2);
    
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*K, 0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE, K*N, 0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE, M*N*2, 0);
    
    printf("Expected: C[m][n] = m * K = m * %d\n", K);
    
    auto run = kernel((unsigned)3, bo_instr, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, M*N*2, 0);
    
    int correct = 0, wrong = 0;
    for (int m = 0; m < M; m++) {
        if (c[m*N] == K*m) correct++; else { wrong++; if (wrong <= 5) printf("Row %d: got %d expected %d\n", m, c[m*N], K*m); }
    }
    printf("Correct: %d/%d, Wrong: %d\n", correct, M, wrong);
    return wrong > 0 ? 1 : 0;
}
