#include "alloc.h"

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}
// 重写CPU申请内存函数
void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) {
        return nullptr;
    }
    void* data = malloc(byte_size);
    return data;
}
// 重写CPU内存释放函数
void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr) {
        free(ptr);
    }
    return;
}
// 初始化全局唯一CPU内存管理器实例指针
std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}