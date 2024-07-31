#include "auto_engine/cuda/info.h"
#include "auto_engine/cuda/kernel.cuh"

namespace cuda {

void init() {
    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties_v2(&prop, GPU_DEVICE_ID), "get_device_prop")
    LOG(INFO) << "\ncuda info:\ngrid x: " << prop.maxGridSize[0] 
        << "\n grid y: " << prop.maxGridSize[1]
        << "\n grid z: " << prop.maxGridSize[2]
        << ".\n thread per block: " << prop.maxThreadsPerBlock;
    LOG(INFO) << "local cuda config:\ngrid x: " << grid_cnt_x()
        << "\n grid y: " << grid_cnt_y()
        << "\n grid z: " << grid_cnt_z()
        << "\n thread per block: " << tcnt_per_block();
    // 调用一次cuda free初始化cuda
    cudaFree(0);
}

}