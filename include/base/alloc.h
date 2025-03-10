#ifndef _ALLOC_H_
#define _ALLOC_H_
#include <map>
#include <memory>
#include "base.h"

namespace base {
// 拷贝类型
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

// 内存分配基类
class DeviceAllocator {
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual DeviceType device_type() const { return device_type_; }

    virtual void release(void* ptr) const = 0;

    virtual void* allocate(size_t byte_size) const = 0;

    virtual void memcpy(const void* src_ptr, void* dst_ptr, size_t byte_size,
                        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                        bool need_sync = false) const;
    
    virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);
private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// CPU内存分配器
class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

// CUDA内存缓冲区域
struct CudaMemoryBuffer {
    void* data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;
    CudaMemoryBuffer(void* data, size_t byte_size, bool busy) : data(data), byte_size(byte_size), busy(busy) {}
};

// CUDA内存分配区域
class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;

private:
    // mutable: 被mutable修饰的变量，将永远处于可变的状态，即使在一个const函数中。
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

// 全局唯一CPU资源分配器，确保COUDeviceAllocator的实例在程序中只有一个，并且是全局共享的
class CPUDeviceAllocatorFactory {
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};
// 全局唯一CUDA资源分配器
class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
        if (instance == nullptr) {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }
private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};
}
#endif