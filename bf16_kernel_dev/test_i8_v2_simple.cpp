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
    int M=128, K=1024, N=1024;
    auto device = xrt::device(0);
    auto xc = xrt::xclbin(std::string("/home/bcloud/npu-sandbox/npu-infer/build/int8/final_i8_KV_v2.xclbin"));
    device.register_xclbin(xc);
    auto hw = xrt::hw_context(device, xc.get_uuid());
    auto kernel = xrt::kernel(hw, "MLIR_AIE");
    FILE* f = fopen("/home/bcloud/npu-sandbox/npu-infer/build/int8/insts_i8_KV_v2.txt","rb");
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint32_t> instr(sz/4);
    fread(instr.data(),4,instr.size(),f); fclose(f);
    auto bo_instr=xrt::bo(device,instr.size()*4,XCL_BO_FLAGS_CACHEABLE,kernel.group_id(1));
    auto bo_a=xrt::bo(device,M*K,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(3));
    auto bo_b=xrt::bo(device,K*N,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(4));
    auto bo_c=xrt::bo(device,M*N*2,XRT_BO_FLAGS_HOST_ONLY,kernel.group_id(5));
    memcpy(bo_instr.map(),instr.data(),instr.size()*4);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE,instr.size()*4,0);
    
    // Test: A = row value, B = 1 (simple pattern)
    auto a = (int8_t*)bo_a.map();
    auto b = (int8_t*)bo_b.map();
    auto c = (int16_t*)bo_c.map();
    for(int i=0;i<M*K;i++) a[i] = i/K;  // row 0=0, row 1=1, ...
    memset(b,1,K*N);
    memset(c,0,M*N*2);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE,M*K,0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE,K*N,0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE,M*N*2,0);
    auto run=kernel((unsigned)3,bo_instr,(unsigned)instr.size(),bo_a,bo_b,bo_c);
    run.wait(); bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE,M*N*2,0);
    
    printf("Row-pattern test (A[m][k]=m, B=1):\n");
    for(int m=0;m<8;m++) printf("  C[m=%-2d][0]=%5d (expected %d)\n",m,c[m*N],m*1024);
    printf("  ...\n");
    for(int m=32;m<36;m++) printf("  C[m=%-2d][0]=%5d (expected %d)\n",m,c[m*N],m*1024);
    
    // Second test: random small values  
    printf("\nRandom small test:\n");
    srand(42);
    for(int i=0;i<M*K;i++) a[i] = (int8_t)(rand()%10-5);
    for(int i=0;i<K*N;i++) b[i] = (int8_t)(rand()%10-5);
    memset(c,0,M*N*2);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE,M*K,0);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE,K*N,0);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE,M*N*2,0);
    run=kernel((unsigned)3,bo_instr,(unsigned)instr.size(),bo_a,bo_b,bo_c);
    run.wait(); bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE,M*N*2,0);
    
    // Reference
    int16_t ref[128*1024]={0};
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){
        int32_t s=0; for(int k=0;k<K;k++) s+=(int32_t)a[m*K+k]*(int32_t)b[k*N+n]; ref[m*N+n]=(int16_t)s;
    }
    int bad=0;
    for(int i=0;i<std::min(M*N,10000);i++) if(abs((int)c[i]-(int)ref[i])>1) {bad++; if(bad<=3) printf("  [%d] NPU=%d REF=%d\n",i,c[i],ref[i]);}
    printf("  Errors (>1 diff): %d / %d\n",bad,std::min(M*N,10000));
}
