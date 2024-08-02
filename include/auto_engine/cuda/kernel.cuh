#ifndef CUDA_KERNEL_FUNC_H
#define CUDA_KERNEL_FUNC_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/exit_code.h"
#include "auto_engine/config/config.h"
#include "auto_engine/cuda/info.h"
#include "glog/logging.h"

#define CHECK_CUDA_CALL(call, func_name) \
{ \
    auto err = call; \
    if (err != cudaSuccess) { \
        LOG(ERROR) << __FUNCTION__ << " call " << func_name << " err: " << cudaGetErrorString(err); \
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

// 最后一个维度被改变
template<typename T>
__global__ void permute_l(const T* src, T* dst, const u32* strides, const u32* permute_strides, u32 dim_cnt, u32 tsize, const u32* pl) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];

    u32 d = pl[dim_cnt - 1];

    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x;
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;
    if (lindex < strides[d] && uindex < tsize / strides[d]) {
        u32 index = 0;
        for (u32 i = 0; i <= d; i++) {
            auto stride = strides[i] / strides[d];
            u32 dim_index = uindex / stride;
            uindex = uindex % stride;
            index += dim_index * strides[i];
        }
        for (u32 i = d + 1; i < dim_cnt; i++) {
            u32 dim_index = lindex / strides[i];
            lindex = lindex % strides[i];
            index += dim_index * strides[i];
        }

        // 赋值给共享内存
        shared_mem[threadIdx.y][threadIdx.x] = src[index];
    }

    __syncthreads();

    lindex = threadIdx.y + blockDim.x * blockIdx.x;
    uindex = threadIdx.x + blockDim.y * blockIdx.y;
    if (lindex >= strides[d]) {return;}
    if (uindex >= tsize / strides[d]) {return;}

    // 计算维度下标
    u32 output_index = 0;
    for (u32 i = 0; i <= d; i++) {
        auto stride = strides[i] / strides[d];
        u32 dim_index = uindex / stride;
        uindex = uindex % stride;
        output_index += dim_index * permute_strides[pl[i]];
    }
    for (u32 i = d + 1; i < dim_cnt; i++) {
        u32 dim_index = lindex / strides[i];
        lindex = lindex % strides[i];
        output_index += dim_index * permute_strides[pl[i]];
    }

    // 从共享内存读取
    dst[output_index] = shared_mem[threadIdx.x][threadIdx.y];
}

// 最后一个维度不改变
template<typename T>
__global__ void permute_nl(const T* src, T* dst, const u32* strides, const u32* permute_strides, u32 dim_cnt, u32 tsize, const u32* pl) {
    u32 idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= tsize) {return;}

    u32 index = idx;
    u32 output_index = 0;
    for (u32 i = 0; i < dim_cnt; i++) {
        u32 dim_index = index / strides[i];
        index = index % strides[i];
        output_index += dim_index * permute_strides[pl[i]];
    }

    dst[output_index] = src[idx];
}

// sum最后一个维度。sdv是最后一个维度的值
template<typename T>
__global__ void sum_l(const T* src, T* dst, u32 tsize, u32 sdv) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];
    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x;
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;

    if (lindex < sdv && uindex < tsize / sdv) {
        shared_mem[threadIdx.y][threadIdx.x] = src[uindex * sdv + lindex];
    } else {
        shared_mem[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (u32 i = blockDim.x / 2; i > 0; i = i >> 1) {
        if (threadIdx.x < i) {
            shared_mem[threadIdx.y][threadIdx.x] += shared_mem[threadIdx.y][threadIdx.x + i];
        }
        __syncthreads();
    }

    lindex = threadIdx.y + blockDim.x * blockIdx.x;
    uindex = threadIdx.x + blockDim.y * blockIdx.y;

    if (lindex >= sdv || uindex >= tsize / sdv) {return;}

    if (threadIdx.y == 0) {
        atomicAdd(dst + uindex, shared_mem[threadIdx.x][0]);
    }
}

// expand最后一个维度。sdv是最后一个维度的值
template<typename T>
__global__ void expand_l(const T* src, T* dst, u32 tsize, u32 svd) {
    u32 lindex = threadIdx.x + blockDim.x * blockIdx.x;
    u32 uindex = threadIdx.y + blockDim.y * blockIdx.y;
    if (uindex >= tsize / svd || lindex >= svd) {return;}
    dst[uindex * svd + lindex] = src[uindex];
}

// sum除最后一个维度以外的维度。svd该维度的值，stride该维度的步长值
template<typename T>
__global__ void sum_nl(const T* src, T* dst, u32 tsize, u32 svd, u32 stride) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block()];
    u32 lindex = threadIdx.y + blockDim.y * blockIdx.y;
    u32 index = threadIdx.z + blockDim.z * blockIdx.z;
    u32 uindex = blockIdx.x;
    
    // 以2*2*3张量,d=1为例，strides=[6, 3, 1], uindex最大取2，lindex最大取3, index最大取2
    if (lindex >= stride) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}
    if (index >= svd) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}
    if (uindex >= tsize / stride / svd) {shared_mem[threadIdx.z][threadIdx.y] = 0; return;}

    shared_mem[threadIdx.z][threadIdx.y] = src[uindex * stride * svd + index * stride + lindex];

    __syncthreads();

    for (u32 i = blockDim.z / 2; i > 0; i = i >> 1) {
        if (threadIdx.z < i) {
            shared_mem[threadIdx.z][threadIdx.y] += shared_mem[threadIdx.z + i][threadIdx.y];
        }
        __syncthreads();
    }

    if (threadIdx.z == 0) {
        atomicAdd(dst + uindex * stride + lindex, shared_mem[0][threadIdx.y]);
    }
}

// expand除最后一个维度以外的维度。svd该维度的值，stride该维度的步长值
template<typename T>
__global__ void expand_nl(const T* src, T* dst, u32 tsize, u32 svd, u32 stride) {
    u32 lindex = threadIdx.y + blockDim.y * blockIdx.y;
    u32 index = threadIdx.z + blockDim.z * blockIdx.z;
    u32 uindex = blockIdx.x;
    if (lindex >= stride) {return;}
    if (index >= svd) {return;}
    if (uindex >= tsize / stride / svd) {return;}
    dst[uindex * stride * svd + index * stride + lindex] = src[uindex * stride + lindex];
}

template<typename T>
__global__ void sum(const T* src, u32 size, T* dst) {
    __shared__ T shared_mem[cuda::tcnt_per_block()];
    
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 

    if (index >= size) {shared_mem[threadIdx.x] = 0; return;}
    shared_mem[threadIdx.x] = src[index];

    __syncthreads();

    for (u32 i = blockDim.x / 2; i > 0; i = i >> 1) {
        if (threadIdx.x < i) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
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
__global__ void cat(const T* src, const u32* src_dims, const u32* src_strides,
    T* dst, const u32* dst_dims, const u32* dst_strides, 
    u32 dim_cnt, u32 d, u32 d_offset) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= src_strides[0] * src_dims[0]) {return;}
    
    u32 dst_index = 0;
    u32 ridx = idx;
    for (u32 i = 0; i < dim_cnt; i++) {
        auto dim_index = ridx / src_strides[i];
        if (i == d) {
            dim_index += d_offset;
        }
        dst_index += dim_index * dst_strides[i];
        ridx = ridx % src_strides[i];
    }

    dst[dst_index] = src[idx];
}

template<typename T>
__global__ void cat(const T** srcs, const u32** srcs_dims, const u32** srcs_strides, u32 src_cnt,
    T* dst, const u32* dst_dims, const u32* dst_strides, 
    u32 dim_cnt, u32 d) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= dst_strides[0] * dst_dims[0]) {return;}
    
    u32 indexs[MAX_TENSOR_DIM_CNT];
    u32 src_index = 0;

    u32 ridx = idx;
    for (u32 i = 0; i < dim_cnt; i++) {
        auto dim_index = ridx / dst_strides[i];
        if (i == d) {
            for (u32 j = 0; j < src_cnt; j++) {
                if (dim_index < srcs_dims[j][i]) {
                    src_index = j;
                    break;
                }
                dim_index -= srcs_dims[j][i];
            }
        }
        indexs[i] = dim_index;
        ridx = ridx % dst_strides[i];
    }

    u32 iidx = 0;
    for (u32 i = 0; i < dim_cnt; i++) {
        iidx += indexs[i] * srcs_strides[src_index][i];
    }

    dst[idx] = srcs[src_index][iidx];
}

template<typename T>
__global__ void split(const T* src, const u32* src_dims, const u32* src_strides,
    T* dst, const u32* dst_dims, const u32* dst_strides, 
    u32 dim_cnt, u32 d, u32 d_offset) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= dst_strides[0] * dst_dims[0]) {return;}
    
    u32 src_index = 0;
    u32 ridx = idx;
    for (u32 i = 0; i < dim_cnt; i++) {
        auto dim_index = ridx / dst_strides[i];
        if (i == d) {
            dim_index += d_offset;
        }
        src_index += dim_index * src_strides[i];
        ridx = ridx % dst_strides[i];
    }

    dst[idx] = src[src_index];
}

template<typename T>
__global__ void split(const T* src, const u32* src_dims, const u32* src_strides, 
    T** dsts, const u32** dsts_dims, const u32** dsts_strides, u32 dst_cnt, 
    u32 dim_cnt, u32 d) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= src_strides[0] * src_dims[0]) {return;}
    
    u32 indexs[MAX_TENSOR_DIM_CNT];
    u32 dst_index = 0;

    u32 ridx = idx;
    for (u32 i = 0; i < dim_cnt; i++) {
        auto dim_index = ridx / src_strides[i];
        if (i == d) {
            for (u32 j = 0; j < dst_cnt; j++) {
                if (dim_index < dsts_dims[j][i]) {
                    dst_index = j;
                    break;
                }
                dim_index -= dsts_dims[j][i];
            }
        }
        indexs[i] = dim_index;
        ridx = ridx % src_strides[i];
    }

    u32 didx = 0;
    for (u32 i = 0; i < dim_cnt; i++) {
        didx += indexs[i] * dsts_strides[dst_index][i];
    }

    dsts[dst_index][didx] = src[idx];
}

// 仅一维展开成二维
template<typename T>
__global__ void one_hot(const f64* src, u32 size, T* dst, u32 classes, u32* err_occur, u32* err_index) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size * classes) {return;}
    
    u32 src_index = index / classes;
    u32 v = round(src[src_index]);
    if (v >= classes) {
        *err_occur = 1;
        *err_index = src_index;
        return;
    }
    if (v == index % classes) {
        dst[index] = 1;    
    } else {
        dst[index] = 0;
    }
}


}

#endif 