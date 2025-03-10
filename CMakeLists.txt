cmake_minimum_required(VERSION 3.10)
project(TestEncode)

set(CMAKE_CXX_STANDARD 20)

# 头文件路径
include_directories(
    include 
    include/base 
    include/op 
    include/tensor 
    /usr/local/cuda-11.8/include
    source/op/kernels
)

# 库路径
link_directories(/usr/local/cuda-11.8/lib64 /usr/local/lib)

# 源文件
file(GLOB_RECURSE SOURCES
    source/op/encode.cpp
    source/op/layer.cpp
    source/tensor/tensor.cpp
    source/base/*.cpp
    # source/op/*.cpp
    # source/op/kernels/*.cpp
)

# 生成可执行文件
add_executable(test_encode ${SOURCES})

# 链接库
target_link_libraries(test_encode
    cudart
    armadillo
    glog
    sentencepiece
)