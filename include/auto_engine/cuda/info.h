#ifndef CUDA_INFO_H
#define CUDA_INFO_H

#define GPU_DEVICE_ID 0
// #define GPU_DEBUG_MODE // 缩减gpu并行度

namespace cuda {

#ifdef GPU_DEBUG_MODE
#define GRID_CNT_X 1024
#define GRID_CNT_Y 128
#define GRID_CNT_Z 128
#define THREAD_CNT_PER_BLOCK 4
#define SQRT_THREAD_CNT_PER_BLOCK 2
#else
#define GRID_CNT_X 2147483647
#define GRID_CNT_Y 65535
#define GRID_CNT_Z 65535
#define THREAD_CNT_PER_BLOCK 1024
#define SQRT_THREAD_CNT_PER_BLOCK 32
#endif

void init();
constexpr int grid_cnt_x() {return GRID_CNT_X;}
constexpr int grid_cnt_y() {return GRID_CNT_Y;}
constexpr int grid_cnt_z() {return GRID_CNT_Z;}
constexpr int tcnt_per_block() {return THREAD_CNT_PER_BLOCK;}
constexpr int sqrt_tcnt_per_block() {return SQRT_THREAD_CNT_PER_BLOCK;}

}


#endif