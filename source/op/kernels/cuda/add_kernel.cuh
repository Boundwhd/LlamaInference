#ifndef _ADD_KERNEL_CUH_
#define _ADD_KERNEL_CUH_
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream = nullptr);
}  
#endif  
