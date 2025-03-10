#ifndef _SCALE_SUM_KERNEL_H_
#define _SCALE_SUM_KERNEL_H_
#include "tensor.h"
namespace kernel {
void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                          const tensor::Tensor& output, int t, int d, int stride,
                          void* stream = nullptr);
}
#endif