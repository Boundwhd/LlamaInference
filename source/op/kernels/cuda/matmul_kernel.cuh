#ifndef _MATMUL_KERNEL_CUH_
#define _MATMUL_KERNEL_CUH_
#include "tensor.h"
#include "cuda_config.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
    const tensor::Tensor& output, float scale = 1.f, const CudaConfig* config = nullptr);
    
void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
    const tensor::Tensor& output, int32_t group_size, const tensor::Tensor& scale, const CudaConfig* config = nullptr);
}
#endif