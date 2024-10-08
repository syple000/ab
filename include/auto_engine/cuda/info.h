#ifndef CUDA_INFO_H
#define CUDA_INFO_H

#include "auto_engine/base/basic_types.h"
#define GPU_DEVICE_ID 0
// #define GPU_DEBUG_MODE // 缩减gpu并行度

namespace cuda {

#ifdef GPU_DEBUG_MODE
#define GRID_CNT_X 1024
#define GRID_CNT_Y 4
#define GRID_CNT_Z 4
#define THREAD_CNT_PER_BLOCK 1
#define SQRT_THREAD_CNT_PER_BLOCK 1
#else
#define GRID_CNT_X 2147483647
#define GRID_CNT_Y 65535
#define GRID_CNT_Z 65535
#define THREAD_CNT_PER_BLOCK 1024
#define SQRT_THREAD_CNT_PER_BLOCK 32
#endif

void init();
constexpr u32 grid_cnt_x() {return GRID_CNT_X;}
constexpr u32 grid_cnt_y() {return GRID_CNT_Y;}
constexpr u32 grid_cnt_z() {return GRID_CNT_Z;}
constexpr u32 tcnt_per_block() {return THREAD_CNT_PER_BLOCK;}
constexpr u32 sqrt_tcnt_per_block() {return SQRT_THREAD_CNT_PER_BLOCK;}

}


#endif