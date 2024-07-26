#ifndef CUDA_KERNEL_FUNC_H
#define CUDA_KERNEL_FUNC_H

#include "auto_engine/base/exit_code.h"
#include "auto_engine/cuda/info.h"
#include "glog/logging.h"

#define CHECK_CUDA_CALL(call, func_name) \
{ \
    auto err = call; \
    if (err != cudaSuccess) { \
        LOG(ERROR) << __FUNCTION__ << " call " << func_name << " err: " << err; \
        exit(CUDA_ERR); \
    } \
} \

#define CHECK_CUBLAS_CALL(call, func_name) \
{ \
    auto status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        LOG(ERROR) << __FUNCTION__ << " call " << func_name << " err: " << status; \
        exit(CUBLAS_ERR); \
    } \
} \

namespace cuda_kernel {

// apply仅一维dim，blockDim<32-more> blockDim<(size + 31)/32>

template<typename T>
__device__ void apply(T* data1, const T* data2, int size, T f(const T&, const T&)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    data1[index] = f(data1[index], data2[index]);
} 

template<typename T>
__device__ void apply(T* data, int size, T f(const T&)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    data[index] = f(data[index]);
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

template<typename T>
__global__ void apply_sin(T* data, int size) {return apply<T>(data, size, sin);}

template<typename T>
__global__ void apply_cos(T* data, int size) {return apply<T>(data, size, cos);}

template<typename T>
__global__ void apply_log(T* data, int size) {return apply<T>(data, size, log);}

template<typename T>
__global__ void apply_add(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, add);}

template<typename T>
__global__ void apply_sub(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, sub);}

template<typename T>
__global__ void apply_mul(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, mul);}

template<typename T>
__global__ void apply_div(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, div);}

template<typename T>
__global__ void apply_neg(T* data, int size) {return apply<T>(data, size, neg);}

template<typename T>
__global__ void apply_pow(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, pow);}

template<typename T> // 实现的不好
__global__ void apply_sum(const T* src, int src_size, T* dst, int dst_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= dst_size) {return;}
    int res = 0;
    int i = index;
    while (i < src_size) {
        res += src[i];
        i += dst_size;
    }
    dst[index] = res;
}

// 三维，第一&二维是矩阵行列，第三维是矩阵个数。blockDim<TILE_DIM, TILE_DIM>(根据每一个block线程个数确认)，grimDim<row, col, matrix cnt>
template<typename T>
__global__ void transpose(const T* srcs, T* dsts, int row, int col, int size) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1]; // 一个线程块内的共享内存，对应threadIdx.x * threadIdx.y * threadIdx.z

    int matrix_index = blockIdx.z;
    if (matrix_index >= size) {return;}
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row || j >= col) {return;}
    shared_mem[threadIdx.x][threadIdx.y] = srcs[matrix_index * row * col + i * col + j];

    __syncthreads();

    dsts[matrix_index * row * col + j * row + i] = shared_mem[threadIdx.x][threadIdx.y];
}

}

#endif 