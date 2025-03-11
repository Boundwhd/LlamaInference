#ifndef _RAW_MODEL_DATA_H_
#define _RAW_MODEL_DATA_H_
#include <cstddef>
#include <cstdint>

namespace model {
struct RawModelData {
    ~RawModelData();
    int32_t fd = -1;
    size_t file_size = 0;
    void* data = nullptr;
    void* weight_data = nullptr;
    virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
    const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData {
    const void* weight(size_t offset) const override;
};
}
#endif