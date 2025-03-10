#include <iostream>
#include <unordered_dense.h>  // 引入 unordered_dense 库

int main() {
    // 定义一个 unordered_dense::map
    ankerl::unordered_dense::map<std::string, int> map;

    // 插入数据
    map["apple"] = 1;
    map["banana"] = 2;
    map["cherry"] = 3;

    // 查找数据
    if (map.contains("banana")) {
        std::cout << "banana: " << map["banana"] << std::endl;
    }

    // 遍历数据
    for (const auto& [key, value] : map) {
        std::cout << key << ": " << value << std::endl;
    }

    return 0;
}