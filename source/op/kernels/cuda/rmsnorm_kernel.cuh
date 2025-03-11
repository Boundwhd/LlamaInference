#ifndef _RMSNORM_KERNEL_CUH_
#define _RMSNORM_KERNEL_CUH_
#include "tensor.h"
namespace kernel {
void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
    const tensor::Tensor& output, void* stream = nullptr);
}

#endif