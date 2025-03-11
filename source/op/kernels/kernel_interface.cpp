#include "base.h"
#include "add_kernel.h"
#include "emb_kernel.h"
#include "matmul_kernel.h"
#include "mha_kernel.h"
#include "rmsnorm_kernel.h"
#include "rope_kernel.h"
#include "scale_kernel.h"
#include "scale_sum_kernel.h"
#include "softmax_kernel.h"
#include "swiglu_kernel.h"
#include "add_kernel.cuh"
#include "emb_kernel.cuh"
#include "matmul_kernel.cuh"
#include "mha_kernel.cuh"
#include "rmsnorm_kernel.cuh"
#include "rope_kernel.cuh"
#include "swiglu_kernel.cuh"
#include "kernel_interface.h"

namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
    }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return emb_kernel_normal;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return emb_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an embedding kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return matmul_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return matmul_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCUDA) {
        return matmul_kernel_cu_qint8;
    } else {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return mha_kernel;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return mha_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an mha kernel.";
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rope_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return rope_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return softmax_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get an softmax kernel.";
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return swiglu_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return swiglu_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
        return nullptr;
    }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_sum_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
        return nullptr;
    }
}

}
