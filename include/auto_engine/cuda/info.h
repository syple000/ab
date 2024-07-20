#ifndef CUDA_INFO_H
#define CUDA_INFO_H

#define GPU_DEVICE_ID 0

namespace cuda {

#define GRID_CNT_X 2147483647
#define GRID_CNT_Y 65535
#define GRID_CNT_Z 65535
#define THREAD_CNT_PER_BLOCK 1024
#define SQRT_THREAD_CNT_PER_BLOCK 32

void init();
constexpr int grid_cnt_x() {return GRID_CNT_X;}
constexpr int grid_cnt_y() {return GRID_CNT_Y;}
constexpr int grid_cnt_z() {return GRID_CNT_Z;}
constexpr int tcnt_per_block() {return THREAD_CNT_PER_BLOCK;}
constexpr int sqrt_tcnt_per_block() {return SQRT_THREAD_CNT_PER_BLOCK;}

}


#endif