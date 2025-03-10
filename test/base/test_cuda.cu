#include <iostream>
#include <cuda_runtime.h>

int main() {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);

    if (state == cudaSuccess) {
        std::cout << "Current GPU device ID: " << id << std::endl;
    } else {
        std::cerr << "Failed to get device ID: " << cudaGetErrorString(state) << std::endl;
    }

    return 0;
}