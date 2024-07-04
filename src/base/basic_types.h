#ifndef BASE_BASIC_TYPES_H
#define BASE_BASIC_TYPES_H

#include <cstdint>

#define EPSILON 1e-7

static_assert(sizeof(double) == 8, "double size != 8");

typedef uint64_t size64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef double f64; 

#endif