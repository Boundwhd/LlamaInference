#ifndef _CUDA_CONFIG_H_
#define _CUDA_CONFIG_H_
#include <cuda_runtime.h>
namespace kernel {
struct CudaConfig {
    cudaStream_t stream = nullptr;
    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};
}
#endif