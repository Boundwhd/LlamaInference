#include "tick.h"

int main() {
    TICK(my_timer);  
    // 一些耗时代码
    for (int i = 0; i < 1000000; ++i) {}
    TOCK(my_timer);  
    return 0;
}

