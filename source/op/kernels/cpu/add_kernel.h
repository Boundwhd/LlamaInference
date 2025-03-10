#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_
#include "tensor.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}
#endif