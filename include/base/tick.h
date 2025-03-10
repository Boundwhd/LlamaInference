#ifndef _TICK_H_
#define _TICK_H_

#include <chrono>
#include <iostream>

#ifndef __ycm__
/**
 * @brief 用于测量代码块的执行时间
 * ## 变量连接符，用于创建新变量名称
 * 
 */
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                     \
  printf("%s: %lfs\n", #x,                                          \
         std::chrono::duration_cast<std::chrono::duration<double>>( \
             std::chrono::steady_clock::now() - bench_##x)          \
             .count());
#else
#define TICK(x)
#define TOCK(x)
#endif

#endif