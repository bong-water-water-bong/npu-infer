#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main() {
    const char* xclbin_p = "/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV_v2_fixed.xclbin";
    const char* insts_p  = "/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV_v2_fixed.txt";
    int M=128, K=1024, N=1024;

    printf("=== A-Broadcast Isolation Test ===\n\n");

    FILE* f = fopen(insts_p, "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(), 4, instr.size(), f); fclose(f);

    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string(xclbin_p));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");

    // Test: A=all-1s, B=all-1s → C[m][n] should = K
    printf("Test A (all-1s x all-1s = K):\n");
    {
        auto bo_i = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

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
        int ok = 0, bad = 0;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                if (c[m*N+n] == K) ok++; else { bad++; if (bad <= 3) printf("  C[%d][%d]=%d expected %d\n", m, n, c[m*N+n], K); }
        printf("  %d/%d correct, %d wrong\n", ok, M*N, bad);
    }

    // Test: A[m][k]=1, B[k][n] varying by N-slice → shows whether each core sees right N-slice
    printf("\nTest B (A=1, B varies per N-slice):\n");
    {
        auto bo_i = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        memcpy(bo_i.map(), instr.data(), instr.size()*4);
        bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        memset(bo_a.map(), 1, M*K);
        auto* b = (int8_t*)bo_b.map();
        for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                b[k*N + n] = (int8_t)n;  // B[k][n] = n (column index)
        memset(bo_c.map(), 0, M*N*2);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel((unsigned)3, bo_i, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        auto* c = (int16_t*)bo_c.map();
        // Expected: C[m][n] = K * n if B layout correct, else garbled
        printf("  C[0][0:5] = ");
        for (int n = 0; n < 5; n++) printf("%d ", c[n]);
        printf("(expected %d %d %d %d %d)\n", K*0, K*1, K*2, K*3, K*4);
        printf("  C[0][128:133] = ");
        for (int n = 0; n < 5; n++) printf("%d ", c[128+n]);
        printf("(expected all = K*128=%d)\n", K*128);
    }

    // Test: Compare old v2 vs fixed for row-pattern to see if fix changed anything
    printf("\nTest C (A=m, B=1 — compare rows 0..31):\n");
    {
        auto bo_i = xrt::bo(device, instr.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_a = xrt::bo(device, M*K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, K*N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_c = xrt::bo(device, M*N*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        memcpy(bo_i.map(), instr.data(), instr.size()*4);
        bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto* a = (int8_t*)bo_a.map();
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                a[m*K + k] = (int8_t)m;
        memset(bo_b.map(), 1, K*N);
        memset(bo_c.map(), 0, M*N*2);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel((unsigned)3, bo_i, (unsigned)instr.size(), bo_a, bo_b, bo_c);
        run.wait();
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        auto* c = (int16_t*)bo_c.map();
        // Check pattern: with A[m]=m (contiguous 32-row tiles) and B=1, if A-broadcast works,
        // each row m should get m*K. If round-robin (old bug), rows get interleaved.
        // B contamination with B=1 masks itself (all values are 1).
        printf("  Row  |  got  | expected | diff\n");
        for (int m = 0; m < 16; m++) {
            int got = c[m*N];
            int exp = m * K;
            printf("  %3d   | %5d |  %5d   | %+d\n", m, got, exp, got - exp);
        }
        // Also check if ALL rows 0..127 work
        int correct = 0, wrong = 0;
        for (int m = 0; m < M; m++) {
            if (c[m*N] == m*K) correct++;
            else wrong++;
        }
        printf("  Summary: %d correct, %d wrong out of %d rows\n", correct, wrong, M);

        // Show first few wrong rows
        if (wrong > 0) {
            printf("  First wrong rows:\n");
            for (int m = 0, cnt = 0; m < M && cnt < 8; m++) {
                if (c[m*N] != m*K) {
                    printf("    Row %d: got %d, diff=%+d\n", m, c[m*N], c[m*N] - m*K);
                    cnt++;
                }
            }
        }
    }

    return 0;
}
