#ifndef _SWIGLU_KERNEL_CUH_
#define _SWIGLU_KERNEL_CUH_
#include "tensor.h"
namespace kernel {
void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
    const tensor::Tensor& output, void* stream);
}
#endif