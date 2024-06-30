#ifndef UTILS_VEC_UTILS_H
#define UTILS_VEC_UTILS_H

#include <vector>

template<typename... Args>
auto convertToVec(Args... args) {
    return std::vector{args...};
}

#endif