#include "auto_engine/cuda/info.h"
#include "auto_engine/cuda/kernel.cuh"

namespace cuda {

void init() {
    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties_v2(&prop, GPU_DEVICE_ID), "get_device_prop")
//    _tcnt_per_block = prop.maxThreadsPerBlock;
//    _grid_cnt_x = prop.maxGridSize[0];
//    _grid_cnt_y = prop.maxGridSize[1];
//    _grid_cnt_z = prop.maxGridSize[2];
    LOG(INFO) << "grid x: " << prop.maxGridSize[0] 
        << "\n grid y: " << prop.maxGridSize[1]
        << "\n grid z: " << prop.maxGridSize[2]
        << ".\n thread per block: " << prop.maxThreadsPerBlock;
}

}