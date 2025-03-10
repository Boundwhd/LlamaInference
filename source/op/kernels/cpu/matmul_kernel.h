#ifndef _MATMUL_KERNEL_H_
#define _MATMUL_KERNEL_H_
#include "tensor.h"
#include "cuda_config.h"

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const CudaConfig* config = nullptr);
}

#endif 