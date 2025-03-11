#ifndef _SWIGLU_KERNEL_H_
#define _SWIGLU_KERNEL_H_
#include "tensor.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
    const tensor::Tensor& output, void* stream);
}

#endif