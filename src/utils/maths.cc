#include "auto_engine/utils/maths.h"
#include "auto_engine/base/basic_types.h"

namespace utils {

template<>
u32 nextPowerOfTwo(u32 n) {
    if (n <= 1) {return 1;}
    n --;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

template<>
u64 nextPowerOfTwo(u64 n) {
    if (n <= 1) {return 1;}
    n --;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}




}

