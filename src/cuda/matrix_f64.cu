#include "auto_engine/base/slice.h"
#include "auto_engine/cuda/matrix_kernel.cuh"
#include "auto_engine/cuda/matrix_f64.h"
#include "auto_engine/cuda/mem.h"
#include "auto_engine/utils/defer.h"
#include "cublas_v2.h"
#include <climits>
#include <cstdlib>
#include <glog/logging.h>
#include <memory>
#include <sstream>
#include <vector>

namespace cuda {

#define BATCH_COUNT 1

#define CHECK_CUBLAS_CALL(call, func_name) \
{ \
    auto status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        LOG(ERROR) << __FUNCTION__ << "call " << func_name << " err: " << status; \
        return MatrixF64(); \
    } \
} \

#define CHECK_CUDA_CALL(call, func_name) \
{ \
    auto err = call; \
    if (err != cudaSuccess) { \
        LOG(ERROR) << __FUNCTION__ << "call " << func_name << " err: " << err; \
        return MatrixF64(); \
    } \
} \

#define CHECK_INFOS(info_array, period_name) \
{ \
    auto infos = reinterpret_cast<int*>(Mem::device2Host(info_array, BATCH_COUNT * sizeof(int))); \
    if (!infos) { \
        LOG(ERROR) << __FUNCTION__ << "get info err: " << period_name; \
        return MatrixF64(); \
    } \
    utils::Defer free_infos([&infos] {free(reinterpret_cast<void*>(infos));}); \
    for (int i = 0; i < BATCH_COUNT; i++) { \
        if (infos[i] != 0) { \
            LOG(ERROR) << __FUNCTION__ << period_name << " err: " << infos[i]; \
            return MatrixF64(); \
        } \
    } \
} \

MatrixF64::MatrixF64() {} 
MatrixF64::MatrixF64(u32 m, u32 n, const base::Slice<f64>& slice): _m(m), _n(n), _slice(slice) { 
    if (m * n != slice.size()) { 
        LOG(ERROR) << __FUNCTION__ << "matrix size invalid"; 
        _m = 0; 
        _n = 0; 
        _slice = base::Slice<f64>(); 
        return; 
    } 
} 
MatrixF64::MatrixF64(u32 m, u32 n, std::vector<f64>&& vec): _m(m), _n(n) {
    if (m * n != vec.size()) {
        LOG(ERROR) << __FUNCTION__ << "matrix size invalid";
        _m = 0;
        _n = 0;
        return;
    }
    _slice = base::Slice<f64>(std::make_shared<std::vector<f64>>(std::move(vec)));
}
MatrixF64::MatrixF64(const MatrixF64& m) {
    _m = m._m;
    _n = m._n;
    _slice = m._slice;
}
MatrixF64& MatrixF64::operator=(const MatrixF64& m) {
    if (this == &m) {
        return *this;
    }
    _m = m._m;
    _n = m._n;
    _slice = m._slice;
    return *this;
}
MatrixF64::~MatrixF64() {}

bool MatrixF64::operator==(const MatrixF64& m) const {
    if (this == &m) {
        return true;
    }
    if (_m != m._m || _n != m._n) {
        return false;
    }
    return _slice == m._slice; 
}
const base::Slice<f64>& MatrixF64::getSlice() const {
    return _slice;
} 

std::string MatrixF64::toString() const {
    std::stringstream stream;
    stream << "[";
    for (int i = 0; i < _m; i++) {
        for (int j = 0; j < _n; j++) {
            stream << _slice[i * _n + j];
            if (j != _n - 1) {
                stream << ", ";
            }
        }
        if (i != _m - 1) {
            stream << ",\n";
        }
    }
    stream << "]";
    return stream.str();
}

MatrixF64 MatrixF64::transpose() const { 
    cublasHandle_t handle; 
    CHECK_CUBLAS_CALL(cublasCreate(&handle), "create"); 
    utils::Defer destroy_handle([&handle]{cublasDestroy(handle);});
    const f64 alpha = 1.0, beta = 0.0; 
    auto cuda_ret = reinterpret_cast<f64*>(Mem::malloc(_slice.size() * sizeof(f64))); 
    if (!cuda_ret) { 
        LOG(ERROR) << __FUNCTION__ << "cuda mem alloc err"; 
        return MatrixF64(); 
    } 
    utils::Defer free_mem([&cuda_ret] {Mem::free(reinterpret_cast<void*>(cuda_ret));});
    auto cuda_slice = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64)));
    if (!cuda_slice) {
        LOG(ERROR) << __FUNCTION__ << "get cuda slice err";
        return MatrixF64();
    }
    utils::Defer free_cuda_slice([&cuda_slice] {Mem::free(reinterpret_cast<void*>(cuda_slice));});
    CHECK_CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, _m, _n, &alpha, cuda_slice, _n,  
        &beta, cuda_slice, _m, cuda_ret, _m), "geam"); 
    auto ret = reinterpret_cast<f64*>(Mem::device2Host(cuda_ret, _slice.size() * sizeof(f64))); 
    if (!ret) {
        LOG(ERROR) << __FUNCTION__ << "device 2 host err";
        return MatrixF64();
    }
    return MatrixF64(_n, _m, base::Slice<f64>(std::make_shared<std::vector<f64>>(ret, ret+_slice.size())));  
}

MatrixF64 MatrixF64::mmul(MatrixF64& m) const {
    if (_n != m._m) {
        LOG(ERROR) << __FUNCTION__ << "matrix mul unmatch col and row";
        return MatrixF64();
    }
    cublasHandle_t handle; 
    CHECK_CUBLAS_CALL(cublasCreate(&handle), "create"); 
    utils::Defer destroy_handle([&handle]{cublasDestroy(handle);});
    const f64 alpha = 1.0, beta = 0.0; 
    auto size = _m * m._n;
    auto cuda_ret = reinterpret_cast<f64*>(Mem::malloc(size * sizeof(f64)));
    if (!cuda_ret) { 
        LOG(ERROR) << __FUNCTION__ << "cuda mem alloc err"; 
        return MatrixF64(); 
    } 
    utils::Defer free_mem([&cuda_ret] {Mem::free(reinterpret_cast<void*>(cuda_ret));});
    auto cuda_slice = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64)));
    if (!cuda_slice) {
        LOG(ERROR) << __FUNCTION__ << "get cuda slice err";
        return MatrixF64();
    }
    utils::Defer free_cuda_slice([&cuda_slice] {Mem::free(reinterpret_cast<void*>(cuda_slice));});
    CHECK_CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m._n, _m, _n, 
        &alpha, cuda_slice, m._n, cuda_slice, _n, &beta, cuda_ret, m._n), "gemm");
    auto ret = reinterpret_cast<f64*>(Mem::device2Host(cuda_ret, size * sizeof(f64)));
    if (!ret) {
        LOG(ERROR) << __FUNCTION__ << "device 2 host err";
        return MatrixF64();
    }
    return MatrixF64(_m, m._n, base::Slice<f64>(std::make_shared<std::vector<f64>>(ret, ret+size)));  
}

MatrixF64 MatrixF64::inv() const {
    if (_m != _n) {
        LOG(ERROR) << __FUNCTION__ << "only square matrix supports inversion";
        return MatrixF64();
    }
    cublasHandle_t handle; 
    CHECK_CUBLAS_CALL(cublasCreate(&handle), "create"); 
    utils::Defer destroy_handle([&handle]{cublasDestroy(handle);});

    auto cuda_matrix_array = reinterpret_cast<f64**>(Mem::malloc(BATCH_COUNT * sizeof(f64*)));
    if (!cuda_matrix_array) {
        LOG(ERROR) << __FUNCTION__ << "malloc cuda matrix array err";
        return MatrixF64();
    }
    utils::Defer free_cuda_matrix_array([&cuda_matrix_array] {Mem::free(reinterpret_cast<void*>(cuda_matrix_array));}); 

    auto cuda_matrix = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64)));
    if (!cuda_matrix) {
        LOG(ERROR) << __FUNCTION__ << "host 2 device err";
        return MatrixF64();
    }
    utils::Defer free_cuda_matrix([&cuda_matrix] {Mem::free(reinterpret_cast<void*>(cuda_matrix));});
    CHECK_CUDA_CALL(cudaMemcpy(cuda_matrix_array, &cuda_matrix, sizeof(f64*), cudaMemcpyHostToDevice), "cuda_mem_cpy");

    auto pivot_array = reinterpret_cast<int*>(Mem::malloc(_m * BATCH_COUNT * sizeof(int)));
    if (!pivot_array) {
        LOG(ERROR) << __FUNCTION__ << "malloc pivot array err";
        return MatrixF64();
    }
    utils::Defer free_pivot_array([&pivot_array] {Mem::free(reinterpret_cast<void*>(pivot_array));});

    auto info_array = reinterpret_cast<int*>(Mem::malloc(BATCH_COUNT * sizeof(int)));
    if (!info_array) {
        LOG(ERROR) << __FUNCTION__ << "malloc info array err";
        return MatrixF64();
    }
    utils::Defer free_info_array([&info_array] {Mem::free(reinterpret_cast<void*>(info_array));});
    
    CHECK_CUBLAS_CALL(cublasDgetrfBatched(handle, _m, cuda_matrix_array, _m, pivot_array, info_array, BATCH_COUNT), "getrf_batched");
    CHECK_INFOS(info_array, "getrf_batched");

    // 计算矩阵的逆
    auto cuda_carry_array = reinterpret_cast<f64**>(Mem::malloc(BATCH_COUNT * sizeof(f64*)));
    if (!cuda_carry_array) {
        LOG(ERROR) << __FUNCTION__ << "malloc cuda carry array err";
        return MatrixF64();
    }
    utils::Defer free_cuda_carry_array([&cuda_carry_array] {Mem::free(reinterpret_cast<void*>(cuda_carry_array));}); 

    auto cuda_carry = reinterpret_cast<f64*>(Mem::malloc(_slice.size() * sizeof(f64)));
    if (!cuda_carry) {
        LOG(ERROR) << __FUNCTION__ << "malloc cuda carry err";
        return MatrixF64();
    }
    utils::Defer free_cuda_carry([&cuda_carry] {Mem::free(reinterpret_cast<void*>(cuda_carry));});
    CHECK_CUDA_CALL(cudaMemcpy(cuda_carry_array, &cuda_carry, sizeof(f64*), cudaMemcpyHostToDevice), "cuda_mem_cpy");

    CHECK_CUBLAS_CALL(cublasDgetriBatched(handle, _m, cuda_matrix_array, _m, pivot_array, cuda_carry_array, _m, info_array, BATCH_COUNT), "getri_batched");
    CHECK_INFOS(info_array, "getri_batched");

    auto ret = reinterpret_cast<f64*>(Mem::device2Host(cuda_carry, _slice.size() * sizeof(f64)));
    if (!ret) {
        LOG(ERROR) << __FUNCTION__ << "device 2 host err";
        return MatrixF64();
    }
    return MatrixF64(_m, _n, base::Slice<f64>(std::make_shared<std::vector<f64>>(ret, ret+_slice.size())));  
}

#define APPLY_1E(apply_func) \
{ \
    auto dims = cuda_matrix_kernel::MatrixKernelHelper::getDims(_m*_n, 1, 1, 256); \
    auto cuda_slice = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64))); \
    if (!cuda_slice) { \
        LOG(ERROR) << __FUNCTION__ << "host 2 device err"; \
        return MatrixF64(); \
    } \
    utils::Defer free_cuda_slice([&cuda_slice] {Mem::free(reinterpret_cast<void*>(cuda_slice));}); \
    apply_func<f64><<<std::get<0>(dims), std::get<1>(dims)>>>(cuda_slice, _m*_n, 1); \
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply"); \
    auto ret = reinterpret_cast<f64*>(Mem::device2Host(reinterpret_cast<void*>(cuda_slice), _slice.size() * sizeof(f64))); \
    if (!ret) { \
        LOG(ERROR) << __FUNCTION__ << "device 2 host err"; \
        return MatrixF64(); \
    } \
    return MatrixF64(_m, _n, base::Slice<f64>(std::make_shared<std::vector<f64>>(ret, ret+_slice.size())));   \
} \


#define APPLY_2E(apply_func, m)  \
{ \
    if (_m != m._m || _n != m._n) { \
        LOG(ERROR) << __FUNCTION__ << "apply to diff matrixs"; \
        return MatrixF64(); \
    } \
    auto dims = cuda_matrix_kernel::MatrixKernelHelper::getDims(_m*_n, 1, 1, 256); \
    auto cuda_slice1 = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64))); \
    if (!cuda_slice1) { \
        LOG(ERROR) << __FUNCTION__ << "host 2 device err"; \
        return MatrixF64(); \
    } \
    utils::Defer free_cuda_slice1([&cuda_slice1] {Mem::free(reinterpret_cast<void*>(cuda_slice1));}); \
    auto cuda_slice2 = reinterpret_cast<f64*>(Mem::host2Device(m._slice.data(), m._slice.size() * sizeof(f64))); \
    if (!cuda_slice2) { \
        LOG(ERROR) << __FUNCTION__ << "host 2 device err"; \
        return MatrixF64(); \
    } \
    utils::Defer free_cuda_slice2([&cuda_slice2] {Mem::free(reinterpret_cast<void*>(cuda_slice2));}); \
    apply_func<f64><<<std::get<0>(dims), std::get<1>(dims)>>>(cuda_slice1, cuda_slice2, _m*_n, 1); \
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply"); \
    auto ret = reinterpret_cast<f64*>(Mem::device2Host(reinterpret_cast<void*>(cuda_slice1), _slice.size() * sizeof(f64))); \
    if (!ret) { \
        LOG(ERROR) << __FUNCTION__ << "device 2 host err"; \
        return MatrixF64(); \
    } \
    return MatrixF64(_m, _n, base::Slice<f64>(std::make_shared<std::vector<f64>>(ret, ret+_slice.size())));   \
} \

MatrixF64 MatrixF64::sin() const {APPLY_1E(cuda_matrix_kernel::apply_sin);}
MatrixF64 MatrixF64::cos() const {APPLY_1E(cuda_matrix_kernel::apply_cos);}
MatrixF64 MatrixF64::log() const {APPLY_1E(cuda_matrix_kernel::apply_log);}
MatrixF64 MatrixF64::neg() const {APPLY_1E(cuda_matrix_kernel::apply_neg);}
MatrixF64 MatrixF64::add(const MatrixF64& m) const {APPLY_2E(cuda_matrix_kernel::apply_add, m);}
MatrixF64 MatrixF64::sub(const MatrixF64& m) const {APPLY_2E(cuda_matrix_kernel::apply_sub, m);}
MatrixF64 MatrixF64::mul(const MatrixF64& m) const {APPLY_2E(cuda_matrix_kernel::apply_mul, m);}
MatrixF64 MatrixF64::div(const MatrixF64& m) const {APPLY_2E(cuda_matrix_kernel::apply_div, m);}
MatrixF64 MatrixF64::pow(const MatrixF64& m) const {APPLY_2E(cuda_matrix_kernel::apply_pow, m);}

#undef CHECK_CUBLAS_CALL
#undef CHECK_CUDA_CALL
#undef CHECK_INFOS
#undef APPLY_1E
#undef APPLY_2E

}