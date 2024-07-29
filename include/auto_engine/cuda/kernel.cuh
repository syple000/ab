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
__device__ void apply(T* data, T n, int size, T f(const T&, const T&)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    data[index] = f(data[index], n);
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
__global__ void apply_add(T* data, T n, int size) {return apply<T>(data, n, size, add);}

template<typename T>
__global__ void apply_sub(T* data, T n, int size) {return apply<T>(data, n, size, sub);}

template<typename T>
__global__ void apply_mul(T* data, T n, int size) {return apply<T>(data, n, size, mul);}

template<typename T>
__global__ void apply_div(T* data, T n, int size) {return apply<T>(data, n, size, div);}

template<typename T>
__global__ void apply_neg(T* data, int size) {return apply<T>(data, size, neg);}

template<typename T>
__global__ void apply_sign(T* data, int size) {return apply<T>(data, size, sign);}

template<typename T>
__global__ void apply_abs(T* data, int size) {return apply<T>(data, size, abs);}

template<typename T>
__global__ void apply_pow(T* data1, const T* data2, int size) {return apply<T>(data1, data2, size, pow);}

template<typename T>
__global__ void apply_pow(T* data, T n, int size) {return apply<T>(data, n, size, pow);}

template<typename T>
__global__ void sum_along_row(const T* srcs, T* dsts, int row, int col, int size) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];

    int matrix_index = blockIdx.z;
    if (matrix_index >= size) {return;}
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row || j >= col) {
        shared_mem[threadIdx.x][threadIdx.y] = 0;
    } else {
        shared_mem[threadIdx.x][threadIdx.y] = srcs[matrix_index * row * col + i * col + j];
    }

    __syncthreads();

    for (int i = blockDim.y / 2; i > 0; i = i >> 1) {
        if (threadIdx.y < i) {
            shared_mem[threadIdx.x][threadIdx.y] += shared_mem[threadIdx.x][threadIdx.y + i];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        atomicAdd(dsts + matrix_index * row + i, shared_mem[threadIdx.x][0]);
    }
}

template<typename T>
__global__ void expand_along_row(const T* src, T* dst, int row, int col, int size) {
    int matrix_index = blockIdx.z;
    if (matrix_index >= size) {return;}
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row || j >= col) {return;}
    dst[matrix_index * row * col + i * col + j] = src[matrix_index * row + i];
}

template<typename T>
__global__ void sum_along_col(const T* srcs, T* dsts, int row, int col, int size) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()][cuda::sqrt_tcnt_per_block() + 1];

    int matrix_index = blockIdx.z;
    if (matrix_index >= size) {return;}
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row || j >= col) {
        shared_mem[threadIdx.x][threadIdx.y] = 0;
    } else {
        shared_mem[threadIdx.x][threadIdx.y] = srcs[matrix_index * row * col + i * col + j];
    }

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i = i >> 1) {
        if (threadIdx.x < i) {
            shared_mem[threadIdx.x][threadIdx.y] += shared_mem[threadIdx.x + i][threadIdx.y];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(dsts + matrix_index * col + j, shared_mem[0][threadIdx.y]);
    }
}

template<typename T>
__global__ void expand_along_col(const T* src, T* dst, int row, int col, int size) {
    int matrix_index = blockIdx.z;
    if (matrix_index >= size) {return;}
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row || j >= col) {return;}
    dst[matrix_index * row * col + i * col + j] = src[matrix_index * col + j];
}

template<typename T>
__global__ void apply_sum(const T* src, int size, T* dst) {
    __shared__ T shared_mem[cuda::sqrt_tcnt_per_block()];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int shared_mem_index = threadIdx.x;

    if (index < size) {
        shared_mem[shared_mem_index] = src[index];
    } else {
        shared_mem[shared_mem_index] = 0;
    }

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i = i >> 1) {
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
__global__ void apply_expand(const T* src, T* dst, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    dst[index] = src[0];
}

template<typename T>
__global__ void one_hot(const T* src, int size, T* dst, int classes, int* err_index) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= size) {return;}
    // 检查src的数据
    int n = round(src[index]);
    if (n < 0 || n >= classes) {*err_index = index; return;}
    // 清空并赋值
    for (int i = index * classes; i < index * classes + classes; i++) {
        dst[i] = 0;
    }
    int dst_index = index * classes + n;
    dst[dst_index] = 1;
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