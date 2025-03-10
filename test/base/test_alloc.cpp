#include "alloc.h"
#include "buffer.h"
#include <glog/logging.h>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

namespace base {
namespace {

// 测试默认构造和内存分配
void TestDefaultAllocation() {
  auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
  Buffer buffer(1024, cpu_alloc);
  if (buffer.ptr() == nullptr) {
    std::cerr << "TestDefaultAllocation FAILED: ptr is null" << std::endl;
    return;
  }
  if (buffer.byte_size() != 1024) {
    std::cerr << "TestDefaultAllocation FAILED: byte_size mismatch" << std::endl;
    return;
  }
  if (buffer.device_type() != DeviceType::kDeviceCPU) {
    std::cerr << "TestDefaultAllocation FAILED: device_type mismatch" << std::endl;
    return;
  }
  if (buffer.is_external()) {
    std::cerr << "TestDefaultAllocation FAILED: is_external should be false" << std::endl;
    return;
  }
  std::cout << "TestDefaultAllocation PASSED" << std::endl;
}

// 测试外部内存管理
void TestExternalMemory() {
  std::vector<uint8_t> external_mem(1024);
  Buffer buffer(1024, nullptr, external_mem.data(), true);
  if (buffer.ptr() != external_mem.data()) {
    std::cerr << "TestExternalMemory FAILED: ptr mismatch" << std::endl;
    return;
  }
  if (!buffer.is_external()) {
    std::cerr << "TestExternalMemory FAILED: is_external should be true" << std::endl;
    return;
  }
  std::cout << "TestExternalMemory PASSED" << std::endl;
}

// 测试设备间内存拷贝 (CPU <-> CUDA)
void TestCrossDeviceCopy() {
  // 初始化数据
  auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
  Buffer cpu_src(1024, cpu_alloc);
  std::memset(cpu_src.ptr(), 0xAB, 1024); // 填充测试数据

  // 创建CUDA目标缓冲区
  auto cuda_alloc = CUDADeviceAllocatorFactory::get_instance();
  Buffer cuda_dst(1024, cuda_alloc);

  // CPU -> CUDA
  cuda_dst.copy_from(cpu_src);
  cudaDeviceSynchronize(); // 等待拷贝完成

  // CUDA -> CPU验证
  Buffer cpu_dst(1024, cpu_alloc);
  cpu_dst.copy_from(cuda_dst);
  
  // 验证数据一致性
  if (std::memcmp(cpu_src.ptr(), cpu_dst.ptr(), 1024) != 0) {
    std::cerr << "TestCrossDeviceCopy FAILED: data mismatch" << std::endl;
    return;
  }
  std::cout << "TestCrossDeviceCopy PASSED" << std::endl;
}

// 测试大内存分配（超过1MB）
void TestLargeAllocation() {
  auto cuda_alloc = CUDADeviceAllocatorFactory::get_instance();
  const size_t large_size = 2 * 1024 * 1024; // 2MB
  Buffer buffer(large_size, cuda_alloc);
  
  if (buffer.ptr() == nullptr) {
    std::cerr << "TestLargeAllocation FAILED: ptr is null" << std::endl;
    return;
  }
  if (buffer.byte_size() != large_size) {
    std::cerr << "TestLargeAllocation FAILED: byte_size mismatch" << std::endl;
    return;
  }
  
  // 测试内存可写性
  cudaMemset(buffer.ptr(), 0xCD, large_size);
  cudaDeviceSynchronize();
  std::cout << "TestLargeAllocation PASSED" << std::endl;
}

// 测试内存释放
void TestMemoryRelease() {
  auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
  void* ptr = nullptr;
  {
    Buffer buffer(1024, cpu_alloc);
    ptr = buffer.ptr();
    if (ptr == nullptr) {
      std::cerr << "TestMemoryRelease FAILED: ptr is null" << std::endl;
      return;
    }
  } // 析构时自动释放

  // 检测野指针访问（需要确保系统支持捕获非法内存访问）
  bool caught = false;
  try {
    *(int*)ptr = 0xDEADBEEF;
  } catch (...) {
    caught = true;
  }
  if (!caught) {
    std::cerr << "TestMemoryRelease FAILED: did not catch invalid memory access" << std::endl;
    return;
  }
  std::cout << "TestMemoryRelease PASSED" << std::endl;
}

} // namespace
} // namespace base

int main(int argc, char** argv) {
  // 运行测试
  base::TestDefaultAllocation();
  base::TestExternalMemory();
  base::TestCrossDeviceCopy();
  base::TestLargeAllocation();
  base::TestMemoryRelease();

  return 0;
}