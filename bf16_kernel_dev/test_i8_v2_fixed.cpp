#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

static int test_all_ones(xrt::device& dev, xrt::hw_context& hw, xrt::kernel& kernel,
                          std::vector<uint32_t>& instr, int M, int K, int N) {
    printf("--- Test 1: All-1s (C[m][n] should = K) ---\n");
    auto bo_i = xrt::bo(dev, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(dev, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(dev, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(dev, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_i.map(), instr.data(), instr.size()*4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    memset(bo_a.map(), 1, M*K);
    memset(bo_b.map(), 1, K*N);
    memset(bo_c.map(), 0, M*N*2);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel((unsigned)3, bo_i, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* c = (int16_t*)bo_c.map();
    int correct = 0, wrong = 0;
    for (int m = 0; m < M; m++) {
        if (c[m*N] == K) correct++; else {
            wrong++;
            if (wrong <= 5) printf("  Row %d: got %d expected %d\n", m, c[m*N], K);
        }
    }
    // Also check multiple columns
    for (int m = 0; m < M; m++) {
        if (c[m*N + 1] != K) { wrong++; if (wrong <= 10) printf("  Row %d col 1: got %d expected %d\n", m, c[m*N+1], K); }
    }
    printf("  Result: %d/%d rows correct (col 0)\n", correct, M);
    bool pass = (wrong == 0);
    printf("  VERDICT: %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

static int test_row_pattern(xrt::device& dev, xrt::hw_context& hw, xrt::kernel& kernel,
                             std::vector<uint32_t>& instr, int M, int K, int N) {
    printf("--- Test 2: Row-pattern (A[m]=m, B=1, expect C[m][n]=m*K) ---\n");
    auto bo_i = xrt::bo(dev, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(dev, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(dev, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(dev, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_i.map(), instr.data(), instr.size()*4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto* a = (int8_t*)bo_a.map();
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            a[m*K + k] = (int8_t)m;  // A row = row index
    memset(bo_b.map(), 1, K*N);  // B = all 1s
    memset(bo_c.map(), 0, M*N*2);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel((unsigned)3, bo_i, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* c = (int16_t*)bo_c.map();
    int correct = 0, wrong = 0;
    for (int m = 0; m < M; m++) {
        int expected = m * K;
        if (c[m*N] == expected) correct++;
        else { wrong++; if (wrong <= 8) printf("  Row %d: got %d expected %d\n", m, c[m*N], expected); }
    }
    printf("  Result: %d/%d rows correct\n", correct, M);
    // THE KEY TEST: rows 0-31 should ALL be correct if dataReuse works
    int first32 = 0;
    for (int m = 0; m < 32; m++) if (c[m*N] == m*K) first32++;
    printf("  First 32 rows: %d/32 correct (old v2 got this WRONG)\n", first32);
    bool pass = (wrong == 0);
    printf("  VERDICT: %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

static int test_all_8_columns(xrt::device& dev, xrt::hw_context& hw, xrt::kernel& kernel,
                               std::vector<uint32_t>& instr, int M, int K, int N) {
    printf("--- Test 3: 8-column check (verifies no N-slice contamination) ---\n");
    auto bo_i = xrt::bo(dev, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_a = xrt::bo(dev, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(dev, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c = xrt::bo(dev, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    memcpy(bo_i.map(), instr.data(), instr.size()*4);
    bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // A: all 1s
    memset(bo_a.map(), 1, M*K);
    // B: column N_slice = column_index + 1 (so each 128-col slice has unique values)
    auto* b = (int8_t*)bo_b.map();
    for (int n = 0; n < N; n++) {
        int slice = n / 128;  // which N-slice (0-7 for N=1024)
        for (int k = 0; k < K; k++)
            b[k*N + n] = (int8_t)(slice + 1);  // slice 0→1, slice 1→2, ...
    }
    memset(bo_c.map(), 0, M*N*2);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel((unsigned)3, bo_i, (unsigned)instr.size(), bo_a, bo_b, bo_c);
    run.wait();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* c = (int16_t*)bo_c.map();
    // Each column n in slice s should = K * (s+1) since A=1 and B_kn = s+1
    int correct = 0, wrong = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int slice = n / 128;
            int expected = K * (slice + 1);
            if (c[m*N + n] == expected) correct++;
            else { wrong++; if (wrong <= 5) printf("  C[%d][%d] (slice %d): got %d expected %d\n", m, n, slice, c[m*N+n], expected); }
        }
    }
    printf("  Result: %d/%d elements correct\n", correct, M*N);
    bool pass = (wrong == 0);
    printf("  VERDICT: %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    const char* xclbin_p = "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV_v2_fixed.xclbin";
    const char* insts_p  = "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV_v2_fixed.txt";
    int M=128, K=1024, N=1024;

    printf("=== NPU INT8 DataReuse Fix Test ===\n");
    printf("XCLBIN: %s\nINSTS: %s\nM=%d K=%d N=%d\n\n", xclbin_p, insts_p, M, K, N);

    FILE* f = fopen(insts_p, "rb");
    if (!f) { printf("FAIL: cannot open insts file\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(), 4, instr.size(), f); fclose(f);
    printf("Loaded %zu instructions (%ld bytes)\n\n", instr.size(), sz);

    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string(xclbin_p));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");

    int fails = 0;
    fails += test_all_ones(device, hw, kernel, instr, M, K, N);
    fails += test_row_pattern(device, hw, kernel, instr, M, K, N);
    fails += test_all_8_columns(device, hw, kernel, instr, M, K, N);

    printf("=== OVERALL: %d tests failed ===\n", fails);
    return fails;
}
