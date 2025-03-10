#ifndef _RMSNORM_KERNEL_H_
#define _RMSNORM_KERNEL_H_
#include "tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream = nullptr);
}
#endif