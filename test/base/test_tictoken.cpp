#include <iostream>
#include "tiktoken.h"  

int main() {
    // 1. 定义词表（token -> ID 映射）
    ankerl::unordered_dense::map<std::string, int> encoder = {
        {"hello", 1}, {"world", 2}, {" ", 3}, {"!", 4}, {"he", 5}, {"llo", 6}, {"wor", 7}, {"ld", 8}
    };

    // 2. 特殊词表（可以为空）
    ankerl::unordered_dense::map<std::string, int> special_encoder = {};

    // 3. 定义正则表达式模式（用于切分文本）
    std::string pattern = R"(\w+|\s+|.)";  // 这个模式会匹配单词、空格和单个字符

    // 4. 创建 tiktoken 对象
    tiktoken::tiktoken tokenizer(encoder, special_encoder, pattern);

    // 5. 测试编码
    std::string text = "hea!";
    std::vector<int> encoded = tokenizer.encode(text);
    
    std::cout << "Encoded: ";
    for (int token : encoded) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // 6. 测试解码
    std::string decoded = tokenizer.decode(encoded);
    std::cout << "Decoded: " << decoded << std::endl;

    return 0;
}
