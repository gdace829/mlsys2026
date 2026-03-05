#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 这是一个空实现，仅用于满足框架的构建校验
extern "C" __global__ void dummy_kernel() {
    return;
}