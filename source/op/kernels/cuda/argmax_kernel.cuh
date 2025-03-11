#ifndef _ARGMAX_KERNEL_CUH_
#define _ARGMAX_KERNEL_CUH_

#include "tensor.h"
namespace kernel {
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
}
#endif
