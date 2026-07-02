#include <cstdio>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
int main(){
    printf("Testing 12-column xclbin...\n");
    auto d=xrt::device(0);
    auto xc=xrt::xclbin(std::string("/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build/final_12col_test.xclbin"));
    try{
        d.register_xclbin(xc);
        auto hw=xrt::hw_context(d,xc.get_uuid());
        auto k=xrt::kernel(hw,"MLIR_AIE");
        printf("SUCCESS: 12-column xclbin loaded! NPU accepts >8 cols!\n");
    }catch(std::exception&e){
        printf("FAILED: %s\n",e.what());
    }
    return 0;
}
