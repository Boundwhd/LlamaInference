#ifndef _ROPE_KERNEL_CUH_
#define _ROPE_KERNEL_CUH_
#include "tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
    const tensor::Tensor& cos_cache, cudaStream_t stream);
}
#endif