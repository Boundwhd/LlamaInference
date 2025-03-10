#include "base.h"
#include <gtest/gtest.h>  // Google Test 头文件

namespace base {
namespace {

// 测试默认构造函数
TEST(StatusTest, DefaultConstructor) {
    Status status;
    EXPECT_EQ(status.get_err_code(), kSuccess);  // 默认状态码应为 kSuccess
    EXPECT_TRUE(status);  // operator bool() 应返回 true
    EXPECT_EQ(static_cast<int>(status), kSuccess);  // operator int() 应返回 kSuccess
    EXPECT_EQ(status.get_err_msg(), "");  // 默认错误信息应为空
}

// 测试带参数的构造函数
TEST(StatusTest, ParameterizedConstructor) {
    Status status(kInternalError, "Internal error occurred");
    EXPECT_EQ(status.get_err_code(), kInternalError);
    EXPECT_FALSE(status);  // operator bool() 应返回 false
    EXPECT_EQ(static_cast<int>(status), kInternalError);
    EXPECT_EQ(status.get_err_msg(), "Internal error occurred");
}

// 测试赋值运算符
TEST(StatusTest, AssignmentOperator) {
    Status status;
    status = kInvalidArgument;  // 使用 operator=(int)
    EXPECT_EQ(status.get_err_code(), kInvalidArgument);
    EXPECT_FALSE(status);
}

// 测试比较运算符
TEST(StatusTest, ComparisonOperators) {
    Status status(kSuccess);
    EXPECT_TRUE(status == kSuccess);  
    EXPECT_FALSE(status != kSuccess);  
    
    status = kInternalError;
    EXPECT_TRUE(status != kSuccess);
    EXPECT_FALSE(status == kSuccess);
}

// 测试 get_err_code 和 get_err_msg
TEST(StatusTest, GetMethods) {
    Status status(kPathNotValid, "Path is not valid");
    EXPECT_EQ(status.get_err_code(), kPathNotValid);
    EXPECT_EQ(status.get_err_msg(), "Path is not valid");
}

// 测试 set_err_msg
TEST(StatusTest, SetErrorMessage) {
    Status status;
    status.set_err_msg("New error message");
    EXPECT_EQ(status.get_err_msg(), "New error message");
}

// 测试流插入运算符
TEST(StatusTest, StreamInsertionOperator) {
    Status status(kModelParseError, "Model parse error");
    std::ostringstream oss;
    oss << status;  // 使用 operator<<
    EXPECT_EQ(oss.str(), "Model parse error");
}

// 测试 error 命名空间中的工厂函数
TEST(StatusTest, ErrorFactoryFunctions) {
    Status success = error::Success("Operation succeeded");
    EXPECT_EQ(success.get_err_code(), kSuccess);
    EXPECT_EQ(success.get_err_msg(), "Operation succeeded");

    Status notImplemented = error::FunctionNotImplement("Function not implemented");
    EXPECT_EQ(notImplemented.get_err_code(), kFunctionUnImplement);
    EXPECT_EQ(notImplemented.get_err_msg(), "Function not implemented");

    Status pathInvalid = error::PathNotValid("Path is invalid");
    EXPECT_EQ(pathInvalid.get_err_code(), kPathNotValid);
    EXPECT_EQ(pathInvalid.get_err_msg(), "Path is invalid");

    Status modelError = error::ModelParseError("Model parse error");
    EXPECT_EQ(modelError.get_err_code(), kModelParseError);
    EXPECT_EQ(modelError.get_err_msg(), "Model parse error");

    Status internalError = error::InternalError("Internal error");
    EXPECT_EQ(internalError.get_err_code(), kInternalError);
    EXPECT_EQ(internalError.get_err_msg(), "Internal error");

    Status invalidArgument = error::InvalidArgument("Invalid argument");
    EXPECT_EQ(invalidArgument.get_err_code(), kInvalidArgument);
    EXPECT_EQ(invalidArgument.get_err_msg(), "Invalid argument");

    Status keyExists = error::KeyHasExits("Key already exists");
    EXPECT_EQ(keyExists.get_err_code(), kKeyValueHasExist);
    EXPECT_EQ(keyExists.get_err_msg(), "Key already exists");
}

}  // namespace
}  // namespace base