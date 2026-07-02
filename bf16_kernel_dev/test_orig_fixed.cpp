#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main() {
    const char* xclbin_p = "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV_orig_fixed.xclbin";
    const char* insts_p  = "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV_orig_fixed.txt";
    int M=128, K=1024, N=1024;

    printf("=== Original n1_core_i8 Pattern Test ===\n");
    printf("M=%d K=%d N=%d\n\n", M, K, N);

    FILE* f = fopen(insts_p, "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(), 4, instr.size(), f); fclose(f);
    printf("Loaded %zu instructions (%ld bytes)\n\n", instr.size(), sz);

    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string(xclbin_p));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");

    // Test 1: All-1s (baseline)
    printf("Test 1: All-1s (expect C[m][n]=K=%d)\n", K);
    {
        auto bi = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto ba = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bb = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bc = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        memcpy(bi.map(), instr.data(), instr.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(ba.map(), 1, M*K); memset(bb.map(), 1, K*N); memset(bc.map(), 0, M*N*2);
        ba.sync(XCL_BO_SYNC_BO_TO_DEVICE); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE); bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel((unsigned)3, bi, (unsigned)instr.size(), ba, bb, bc);
        run.wait(); bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto* c = (int16_t*)bc.map();

        int ok=0, bad=0;
        for (int i=0; i<M*N; i++) if (c[i]==K) ok++; else { bad++; if (bad<=3) printf("  C[%d][%d]=%d expected %d\n", i/N, i%N, c[i], K); }
        printf("  %d/%d correct, %d wrong\n", ok, M*N, bad);
        printf("  VERDICT: %s\n\n", bad ? "FAIL" : "PASS");
    }

    // Test 2: A=row_index, B=1 (the real test — old v2 FAILED this)
    printf("Test 2: A[m][k]=m, B=1 (expect C[m][n]=m*K)\n");
    {
        auto bi = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto ba = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bb = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bc = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        memcpy(bi.map(), instr.data(), instr.size()*4); bi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto* a = (int8_t*)ba.map();
        for (int m=0; m<M; m++) for (int k=0; k<K; k++) a[m*K+k] = (int8_t)m;
        memset(bb.map(), 1, K*N); memset(bc.map(), 0, M*N*2);
        ba.sync(XCL_BO_SYNC_BO_TO_DEVICE); bb.sync(XCL_BO_SYNC_BO_TO_DEVICE); bc.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel((unsigned)3, bi, (unsigned)instr.size(), ba, bb, bc);
        run.wait(); bc.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto* c = (int16_t*)bc.map();

        printf("  Row  |  got  | expected | diff\n");
        for (int m=0; m<16; m++) printf("  %3d  | %5d |   %5d  | %+d\n", m, c[m*N], m*K, c[m*N]-m*K);
        int ok=0;
        for (int m=0; m<M; m++) if (c[m*N]==m*K) ok++;
        printf("  %d/%d rows correct\n", ok, M);
        printf("  VERDICT: %s\n\n", ok==M ? "PASS" : "FAIL");
    }

    return 0;
}
