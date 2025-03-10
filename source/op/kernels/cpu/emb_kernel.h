#ifndef _EMB_KERNEL_H_
#define _EMB_KERNEL_H_
#include "tensor.h"
namespace kernel {
    void emb_kernel_normal(const tensor::Tensor& input, const tensor::Tensor& weight,
        const tensor::Tensor& output, int32_t vocab_size, void* stream = nullptr);
}
#endif