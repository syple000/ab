#ifndef CUDA_KERNEL_FUNC_H
#define CUDA_KERNEL_FUNC_H

#include "auto_engine/base/basic_types.h"
#include <tuple>

namespace cuda_device_kernel {

template<typename T>
__device__ void apply(T* data1, T* data2, u32 row_cnt, u32 col_cnt, T f(const T&, const T&)) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < row_cnt && col < col_cnt) {
        u32 index = row * col_cnt + col;
        data1[index] = f(data1[index], data2[index]);
    }
} 

template<typename T>
__device__ void apply(T* data, int row_cnt, int col_cnt, T f(const T&)) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < row_cnt && col < col_cnt) {
        u32 index = row * col_cnt + col;
        data[index] = f(data[index]);
    }
}

template<typename T>
inline __device__ T sin(const T& n) {
    return ::sin(n);
}

template<typename T>
inline __device__ T cos(const T& n) {
    return ::cos(n);
}

template<typename T>
inline __device__ T log(const T& n) {
    return ::log(n);
}

template<typename T>
inline __device__ T neg(const T& n) {
    return -n;
}

template<typename T>
inline __device__ T add(const T& n1, const T& n2) {
    return n1 + n2;
}

template<typename T>
inline __device__ T sub(const T& n1, const T& n2) {
    return n1 - n2;
}

template<typename T>
inline __device__ T mul(const T& n1, const T& n2) {
    return n1 * n2;
}

template<typename T>
inline __device__ T div(const T& n1, const T& n2) {
    return n1 / n2;
}

template<typename T>
inline __device__ T pow(const T& n1, const T& n2) {
    return ::pow(n1, n2);
}
}

namespace cuda_matrix_kernel {

class MatrixKernelHelper {
public:
    static std::tuple<dim3, dim3> getDims(u32 row_cnt, u32 col_cnt, u32 block_dimx = 16, u32 block_dimy = 16) {
        auto block_dim = dim3(block_dimx, block_dimy);
        auto grid_dimx = (col_cnt + block_dimx - 1) / block_dimx;
        auto grid_dimy = (row_cnt + block_dimy - 1) / block_dimy;
        return std::tuple<dim3, dim3>(dim3(grid_dimx, grid_dimy), block_dim);
    }
};

template<typename T>
__global__ void apply_sin(T* data, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data, row_cnt, col_cnt, cuda_device_kernel::sin);}

template<typename T>
__global__ void apply_cos(T* data, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data, row_cnt, col_cnt, cuda_device_kernel::cos);}

template<typename T>
__global__ void apply_log(T* data, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data, row_cnt, col_cnt, cuda_device_kernel::log);}

template<typename T>
__global__ void apply_add(T* data1, T* data2, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data1, data2, row_cnt, col_cnt, cuda_device_kernel::add);}

template<typename T>
__global__ void apply_sub(T* data1, T* data2, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data1, data2, row_cnt, col_cnt, cuda_device_kernel::sub);}

template<typename T>
__global__ void apply_mul(T* data1, T* data2, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data1, data2, row_cnt, col_cnt, cuda_device_kernel::mul);}

template<typename T>
__global__ void apply_div(T* data1, T* data2, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data1, data2, row_cnt, col_cnt, cuda_device_kernel::div);}

template<typename T>
__global__ void apply_neg(T* data, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data, row_cnt, col_cnt, cuda_device_kernel::neg);}

template<typename T>
__global__ void apply_pow(T* data1, T* data2, int row_cnt, int col_cnt) {return cuda_device_kernel::apply<T>(data1, data2, row_cnt, col_cnt, cuda_device_kernel::pow);}
}

#endif 