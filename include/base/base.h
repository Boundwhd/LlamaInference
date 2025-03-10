#ifndef _BASE_H_
#define _BASE_H_
#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>
#include <cstdint>
#include <string>
#include <iostream>
// 防止编译器产生“未使用变量”或“未使用表达式”的警告
#define UNUSED(expr) do {(void)(expr);} while (0)

namespace model {
// Llama模型网络层
enum class ModelBufferType {
    kInputTokens = 0,
    kInputEmbeddings = 1,
    kOutputRMSNorm = 2,
    kKeyCache = 3,
    kValueCache = 4,
    kQuery = 5,
    kInputPos = 6,
    kScoreStorage = 7,
    kOutputMHA = 8,
    kAttnOutput = 9,
    kW1Output = 10,
    kW2Output = 11,
    kW3Output = 12,
    kFFNRMSNorm = 13,
    kForwardOutput = 15,
    kForwardOutputCPU = 16,
    kSinCache = 17,
    kCosCache = 18,
};
}

namespace base {
// 设备类型
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

// 模型类型
enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};

// 数据类型
enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

// 返回数据类型字节函数
inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    } else {
        return 0;
    }
}

// 父类（子类继承后实现资源独享）
class NoCopyable {
    protected:
        NoCopyable() = default;
    
        ~NoCopyable() = default;
    
        NoCopyable(const NoCopyable&) = delete;
    
        NoCopyable& operator=(const NoCopyable&) = delete;
    };

// 状态码
enum StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

// 分词算法
enum class TokenizerType {
    kEncodeUnknown = -1,
    kEncodeSpe = 0,
    kEncodeBpe = 1,
};

// 封装状态码和错误信息，用于表示某个操作的结果状态（成功或失败）
class Status {
public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");

    Status(const Status& other) = default;

    Status& operator=(const Status& other) = default;

    Status& operator=(int code);

    bool operator==(int code) const;

    bool operator!=(int code) const;

    operator int() const;

    operator bool() const;

    uint8_t get_err_code() const;

    const std::string& get_err_msg() const;

    void set_err_msg(const std::string& err_msg);

private:
    int code_ = StatusCode::kSuccess;
    std::string message_;
};

//将 Status 对象的状态码和错误信息以特定的格式输出到流中。
std::ostream& operator<<(std::ostream& os, const Status& x);

namespace error {
    #define STATUS_CHECK(call)                                                                 \
        do {                                                                                   \
            const base::Status& status = call;                                                 \
            if (!status) {                                                                     \
                const size_t buf_size = 512;                                                   \
                char buf[buf_size];                                                            \
                snprintf(buf, buf_size - 1,                                                    \
                   "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
                   __LINE__, int(status), status.get_err_msg().c_str());                       \
                LOG(FATAL) << buf;                                                             \
            }                                                                                  \
        } while (0)
    
    Status Success(const std::string& err_msg = "");

    Status FunctionNotImplement(const std::string& err_msg = "");
    
    Status PathNotValid(const std::string& err_msg = "");
    
    Status ModelParseError(const std::string& err_msg = "");
    
    Status InternalError(const std::string& err_msg = "");
    
    Status KeyHasExits(const std::string& err_msg = "");
    
    Status InvalidArgument(const std::string& err_msg = "");
}   
}
#endif