#ifndef _SOFTMAX_KERNEL_H_
#define _SOFTMAX_KERNEL_H_
#include "tensor.h"
namespace kernel {
void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
}
#endif