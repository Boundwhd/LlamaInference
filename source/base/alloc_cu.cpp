#include <cuda_runtime.h>
#include "alloc.h"

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    //获取当前使用 cuda ID
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);

    if (byte_size > 1024 * 1024) {
        auto& big_buffers = big_buffers_map_[id];
        int sel_id = -1;
        for (int i = 0; i < big_buffers.size(); i++) {
            /* 检查缓冲区是否满足条件：*/
            // 1. 缓冲区大小 >= 请求大小
            // 2. 缓冲区未被占用
            // 3. 缓冲区大小与请求大小的差值 < 1MB：既能减少内存浪费，又能保证一定的灵活性。
            if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                // 如果当前缓冲区比之前选择的缓冲区更小，则选择当前缓冲区
                if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                    sel_id = i;
                }
            }
        }
        // 找到后返回这段buffer数据指针
        if (sel_id != -1) {
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        // cuda显存不够，无法申请
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                     "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                     "left on  device.",
                     byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }
    //小内存分配
    auto& cuda_buffers = cuda_buffers_map_[id];
    for (int i = 0; i < cuda_buffers.size(); i++) {
        if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
            cuda_buffers[i].busy = true;
            no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
        return cuda_buffers[i].data;
        }
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
        char buf[256];
        snprintf(buf, 256,
                "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                "left on  device.",
                byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (!ptr) {
        return;
    }
    //检测每个GPU的小内存空闲区域是否大于1GB，如果大于则删除空闲的位置
    cudaError_t state = cudaSuccess;
    for (auto& it : cuda_buffers_map_) {
        if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for (int i = 0; i < cuda_buffers.size(); i++) {
                if (!cuda_buffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cuda_buffers[i].data);
                    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device " << it.first;
                } else {
                    temp.push_back(cuda_buffers[i]);
                }
            }
            cuda_buffers.clear();
            it.second = temp;
            no_busy_cnt_[it.first] = 0;
        }
    }
    // 将buffer块置于不忙碌状态
    for (auto& it : cuda_buffers_map_) {
        auto& cuda_buffers = it.second;
        for (int i = 0; i < cuda_buffers.size(); i++) {
            if (cuda_buffers[i].data == ptr) {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }
        auto& big_buffers = big_buffers_map_[it.first];
        for (int i = 0; i < big_buffers.size(); i++) {
            if (big_buffers[i].data == ptr) {
                big_buffers[i].busy = false;
                return;
            }
        }
    }
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
}