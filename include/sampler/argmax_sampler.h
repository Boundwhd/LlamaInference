#ifndef _ARGMAX_SAMPLER_H_
#define _ARGMAX_SAMPLER_H_
#include "sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
public:
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}
    size_t sample(const float* logits, size_t size, void* stream) override;
};
}

#endif