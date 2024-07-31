#ifndef CUDA_KERNEL_FUNC_H
#define CUDA_KERNEL_FUNC_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/exit_code.h"
#include "auto_engine/cuda/info.h"
#include "glog/logging.h"
#include <__clang_cuda_runtime_wrapper.h>

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
__device__ void apply(T* data1, const T* data2, u32 size, T f(const T&, const T&)) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    data1[index] = f(data1[index], data2[index]);
} 

template<typename T>
__device__ void apply(T* data, T n, u32 size, T f(const T&, const T&)) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    data[index] = f(data[index], n);
}

template<typename T>
__device__ void apply(T* data, u32 size, T f(const T&)) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
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
inline __device__ T sign(const T& n) {
    if (n >= 0) {
        return 1;
    } else {
        return -1;
    }
}

template<typename T>
inline __device__ T abs(const T& n) {
    return ::abs(n);
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
__global__ void apply_sin(T* data, u32 size) {return apply<T>(data, size, sin);}

template<typename T>
__global__ void apply_cos(T* data, u32 size) {return apply<T>(data, size, cos);}

template<typename T>
__global__ void apply_log(T* data, u32 size) {return apply<T>(data, size, log);}

template<typename T>
__global__ void apply_add(T* data1, const T* data2, u32 size) {return apply<T>(data1, data2, size, add);}

template<typename T>
__global__ void apply_sub(T* data1, const T* data2, u32 size) {return apply<T>(data1, data2, size, sub);}

template<typename T>
__global__ void apply_mul(T* data1, const T* data2, u32 size) {return apply<T>(data1, data2, size, mul);}

template<typename T>
__global__ void apply_div(T* data1, const T* data2, u32 size) {return apply<T>(data1, data2, size, div);}

template<typename T>
__global__ void apply_add(T* data, T n, u32 size) {return apply<T>(data, n, size, add);}

template<typename T>
__global__ void apply_sub(T* data, T n, u32 size) {return apply<T>(data, n, size, sub);}

template<typename T>
__global__ void apply_mul(T* data, T n, u32 size) {return apply<T>(data, n, size, mul);}

template<typename T>
__global__ void apply_div(T* data, T n, u32 size) {return apply<T>(data, n, size, div);}

template<typename T>
__global__ void apply_neg(T* data, u32 size) {return apply<T>(data, size, neg);}

template<typename T>
__global__ void apply_sign(T* data, u32 size) {return apply<T>(data, size, sign);}

template<typename T>
__global__ void apply_abs(T* data, u32 size) {return apply<T>(data, size, abs);}

template<typename T>
__global__ void apply_pow(T* data1, const T* data2, u32 size) {return apply<T>(data1, data2, size, pow);}

template<typename T>
__global__ void apply_pow(T* data, T n, u32 size) {return apply<T>(data, n, size, pow);}


/* 重构实现，支持任意tensor维度进行加和&转置 */

// 转置最后一个维度 & 任意其它维度
template<typename T>
__global__ void transpose(const T* src, T* dst, const u32* dims, const u32* strides, const u32* transpose_strides, u32 dim_cnt, u32 d) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];

    // 以2*2*3张量,d=1为例，strides=[6, 3, 1], uindex最大取2*2，lindex最大取3
    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x;
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;
    if (lindex >= strides[d]) {return;}
    if (uindex >= dims[0] * strides[0] / strides[d]) {return;}

    // 计算维度下标
    u32 index = 0;
    u32 output_index = 0;
    for (u32 i = 0; i < d; i++) {
        auto stride = strides[i] / strides[d];
        u32 dim_index = uindex / stride;
        uindex = uindex % stride;
        index += dim_index * strides[i];
        output_index += dim_index * transpose_strides[i];
    }
    for (u32 i = d + 1; i < dim_cnt - 1; i++) {
        u32 dim_index = lindex / strides[i];
        lindex = lindex % strides[i];
        index += dim_index * strides[i];
        output_index += dim_index * transpose_strides[i];
    }
    index += lindex * strides[dim_cnt - 1];
    output_index += uindex * transpose_strides[dim_cnt - 1];
    index += uindex * strides[d];
    output_index += lindex * transpose_strides[d];

    // 赋值给共享内存
    shared_mem[threadIdx.y][threadIdx.x] = src[index];

    __syncthreads();

    dst[output_index] = shared_mem[threadIdx.y][threadIdx.x];
}

// 转置除最后一个维度的任意两个维度
template<typename T>
__global__ void transpose(const T* src, T* dst, const u32* dims, const u32* strides, const u32* transpose_strides, u32 dim_cnt, u32 d1, u32 d2) {
    u32 index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= strides[0] * dims[0]) {return;}

    u32 rindex = index;
    u32 output_index = 0;
    u32 d1index, d2index;
    for (u32 i = 0; i < dim_cnt; i++) {
        u32 dim_index = rindex / strides[i];
        if (i == d1) {
            d1index = dim_index;
        } else if (i == d2) {
            d2index = dim_index;
        } else {
            output_index += dim_index * transpose_strides[i];
        }
        rindex = rindex % strides[i];
    }
    output_index += d1index * transpose_strides[d2];
    output_index += d2index * transpose_strides[d1];

    dst[output_index] = src[index];
}

// sum最后一个维度
template<typename T>
__global__ void sum(const T* src, T* dst, const u32* dims, const u32* strides, u32 dim_cnt) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];
    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x; // 最后一个维度index
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;

    if (lindex >= dims[dim_cnt - 1] || uindex >= strides[0]*dims[0]/dims[dim_cnt - 1]) { shared_mem[threadIdx.y][threadIdx.x] = 0; return;}
    
    shared_mem[threadIdx.y][threadIdx.x] = src[uindex * dims[dim_cnt - 1] + lindex];

    __syncthreads();

    for (u32 i = blockDim.x / 2; i > 0; i = i >> 1) {
        if (threadIdx.x < i) {
            shared_mem[threadIdx.y][threadIdx.x] += shared_mem[threadIdx.y][threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(dst + uindex, shared_mem[threadIdx.y][0]);
    }
}

template<typename T>
__global__ void expand(const T* src, T* dst, const u32* dims, const u32* strides, u32 dim_cnt, u32 expd) {
    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x; // 待扩展维度index
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;
    if (uindex >= strides[0]*dims[0] || lindex >= expd) {return;}
    dst[uindex * expd + lindex] = src[uindex];
}

// sum除最后一个维度以外的维度
template<typename T>
__global__ void sum(const T* src, T* dst, const u32* dims, const u32* strides, u32 dim_cnt, u32 d) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];
    u32 lindex = threadIdx.y + blockDim.y * blockIdx.y;
    u32 index = threadIdx.z + blockDim.z * blockIdx.z;
    u32 uindex = blockIdx.x;
    
    // 以2*2*3张量,d=1为例，strides=[6, 3, 1], uindex最大取2，lindex最大取3, index最大取2
    if (lindex >= strides[d]) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}
    if (index >= dims[d]) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}
    if (uindex >= strides[0] * dims[0] / strides[d] / dims[d]) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}

    shared_mem[threadIdx.z][threadIdx.y] = src[uindex * strides[d] * dims[d] + index * strides[d] + lindex];

    __syncthreads();

    for (u32 i = blockDim.z / 2; i > 0; i = i >> 1) {
        if (threadIdx.z < i) {
            shared_mem[threadIdx.z][threadIdx.y] += shared_mem[threadIdx.z + i][threadIdx.y];
        }
        __syncthreads();
    }

    if (threadIdx.z == 0) {
        atomicAdd(dst + uindex * strides[d] + lindex, shared_mem[0][threadIdx.y]);
    }
}

template<typename T>
__global__ void expand(const T* src, T* dst, const u32* dims, const u32* strides, u32 dim_cnt, u32 d, u32 expd) {
    u32 lindex = threadIdx.y + blockDim.y * blockIdx.y;
    u32 index = threadIdx.z + blockDim.z * blockIdx.z;
    u32 uindex = blockIdx.x;
    // 以2*3张量,d=1为例，strides=[3, 1], uindex最大取2，lindex最大取3, index最大取2
    // 最后得到2*2*3
    if (lindex >= strides[d] * dims[d]) {return;}
    if (index >= expd) {return;}
    if (uindex >= strides[0] * dims[0] / strides[d] / dims[d]) {return;}
    dst[uindex * strides[d] * dims[d] * expd + index * strides[d] * dims[d] + lindex] = src[uindex * strides[d] * dims[d] + lindex];
}

template<typename T>
__global__ void sum(const T* src, u32 size, T* dst) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()];
    
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    u32 shared_mem_index = threadIdx.x;

    if (index >= size) {shared_mem[shared_mem_index] = 0; return;}
    shared_mem[shared_mem_index] = src[index];

    __syncthreads();

    for (u32 i = blockDim.x / 2; i > 0; i = i >> 1) {
        if (shared_mem_index < i) {
            shared_mem[shared_mem_index] += shared_mem[shared_mem_index + i];
        }
        __syncthreads();
    }

    if (shared_mem_index == 0) {
        atomicAdd(dst, shared_mem[0]); 
    }
}

template<typename T>
__global__ void expand(const T* src, T* dst, u32 size) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    dst[index] = src[0];
}

template<typename T>
__global__ void one_hot(const T* src, u32 size, T* dst, u32 classes, u32* err_index) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    // 检查src的数据
    auto n = round(src[index]);
    if ((u32)n >= classes || n < 0) {*err_index = index; return;}
    // 清空并赋值
    for (u32 i = index * classes; i < index * classes + classes; i++) {
        dst[i] = 0;
    }
    u32 dst_index = index * classes + (u32)n;
    dst[dst_index] = 1;
}


}

#endif 