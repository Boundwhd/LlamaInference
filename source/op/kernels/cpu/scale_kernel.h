#ifndef _SCALE_KERNEL_H_
#define _SCALE_KERNEL_H_
#include "tensor.h"
namespace kernel {
void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}
#endif
