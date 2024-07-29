#include "auto_engine/base/exit_code.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/cuda/mem.h"
#include "auto_engine/cuda/tensor.h"
#include "auto_engine/cuda/kernel.cuh"
#include "auto_engine/utils/defer.h"
#include "cublas_v2.h"
#include <cstdlib>
#include <tuple>

namespace cuda {

#define CHECK_MALLOC(call) \
{ \
    auto succ = call; \
    if (!succ) { \
        LOG(ERROR) << __FUNCTION__ << " malloc cuda err"; \
        exit(CUDA_ERR); \
    } \
} \

std::tuple<dim3, dim3> get_apply_dims(int size) {
    int grid_dim = (size + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    return std::tuple(dim3(grid_dim), dim3(sqrt_tcnt_per_block())); 
}

std::tuple<dim3, dim3> get_matrix_dims(int row, int col, int size) {
    int grid_dim_x = (row + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    int grid_dim_y = (col + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    int grid_dim_z = size;
    return std::tuple(dim3(grid_dim_x, grid_dim_y, grid_dim_z), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block()));
}

#define DEFINE_APPLY_1E(fn, T) \
void fn(T* data, int size) { \
    f64* m; \
    CHECK_MALLOC(Mem::malloc((void**)&m, sizeof(T) * size)); \
    utils::Defer free_m([&m]() {Mem::free(m);}); \
    CHECK_CUDA_CALL(cudaMemcpy(m, data, sizeof(T) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d"); \
    auto dims = get_apply_dims(size); \
    cuda_kernel::fn<<<std::get<0>(dims), std::get<1>(dims)>>>(m, size); \
    CHECK_CUDA_CALL(cudaPeekAtLastError(), #fn); \
    CHECK_CUDA_CALL(cudaMemcpy(data, m, sizeof(T) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h"); \
} \

DEFINE_APPLY_1E(apply_sin, f64)
DEFINE_APPLY_1E(apply_cos, f64)
DEFINE_APPLY_1E(apply_log, f64)
DEFINE_APPLY_1E(apply_neg, f64)
DEFINE_APPLY_1E(apply_sign, f64)
DEFINE_APPLY_1E(apply_abs, f64)

#define DEFINE_APPLY_2E(fn, T) \
void fn(T* data1, const T* data2, int size) { \
    f64 *m1, *m2; \
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(T) * size)); \
    utils::Defer free_m1([&m1]() {Mem::free(m1);}); \
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(T) * size)); \
    utils::Defer free_m2([&m2]() {Mem::free(m2);}); \
 \
    CHECK_CUDA_CALL(cudaMemcpy(m1, data1, sizeof(T) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d"); \
    CHECK_CUDA_CALL(cudaMemcpy(m2, data2, sizeof(T) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d"); \
    auto dims = get_apply_dims(size); \
    cuda_kernel::fn<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, size); \
    CHECK_CUDA_CALL(cudaPeekAtLastError(), #fn); \
    CHECK_CUDA_CALL(cudaMemcpy(data1, m1, sizeof(T) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h"); \
} \

#define DEFINE_APPLY_1E_1T(fn, T) \
void fn(T* data, T n, int size) { \
    f64 *m; \
    CHECK_MALLOC(Mem::malloc((void**)&m, sizeof(T) * size)); \
    utils::Defer free_m([&m]() {Mem::free(m);}); \
 \
    CHECK_CUDA_CALL(cudaMemcpy(m, data, sizeof(T) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d"); \
    auto dims = get_apply_dims(size); \
    cuda_kernel::fn<<<std::get<0>(dims), std::get<1>(dims)>>>(m, n, size); \
    CHECK_CUDA_CALL(cudaPeekAtLastError(), #fn); \
    CHECK_CUDA_CALL(cudaMemcpy(data, m, sizeof(T) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h"); \
} \

DEFINE_APPLY_2E(apply_add, f64)
DEFINE_APPLY_2E(apply_sub, f64)
DEFINE_APPLY_2E(apply_mul, f64)
DEFINE_APPLY_2E(apply_div, f64)
DEFINE_APPLY_2E(apply_pow, f64)

DEFINE_APPLY_1E_1T(apply_add, f64)
DEFINE_APPLY_1E_1T(apply_sub, f64)
DEFINE_APPLY_1E_1T(apply_mul, f64)
DEFINE_APPLY_1E_1T(apply_div, f64)
DEFINE_APPLY_1E_1T(apply_pow, f64)

void sum_along_row(const f64* srcs, f64* dsts, int row, int col, int size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * row * col * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * row * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, srcs, sizeof(f64) * row * col * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(m2, 0, sizeof(f64) * row * size), "cuda_memset");
    auto dims = get_matrix_dims(row, col, size);
    cuda_kernel::sum_along_row<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, row, col, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum_along_row");
    CHECK_CUDA_CALL(cudaMemcpy(dsts, m2, sizeof(f64) * row * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand_along_row(const f64* src, f64* dst, int row, int col, int size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * row * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * row * col * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * row * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_matrix_dims(row, col, size);
    cuda_kernel::expand_along_row<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, row, col, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand_along_row");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * row * col * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void sum_along_col(const f64* srcs, f64* dsts, int row, int col, int size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * row * col * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * col * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, srcs, sizeof(f64) * row * col * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(m2, 0, sizeof(f64) * col * size), "cuda_memset");
    auto dims = get_matrix_dims(row, col, size);
    cuda_kernel::sum_along_col<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, row, col, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum_along_col");
    CHECK_CUDA_CALL(cudaMemcpy(dsts, m2, sizeof(f64) * col * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand_along_col(const f64* src, f64* dst, int row, int col, int size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * col * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * row * col * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * col * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_matrix_dims(row, col, size);
    cuda_kernel::expand_along_col<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, row, col, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand_along_col");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * row * col * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void apply_sum(const f64* src, int size, f64* dst) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64)));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(m2, 0, sizeof(f64)), "cuda_memset");
    auto dims = get_apply_dims(size);
    cuda_kernel::apply_sum<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, size, m2);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply_sum");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void apply_expand(const f64* src, f64* dst, int size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64)));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_apply_dims(size);
    cuda_kernel::apply_expand<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply_expand");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

int ont_hot(const f64* src, int size, f64* dst, int classes) {
    int err_index = -1;

    f64 *m1, *m2; int *m3;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * size * classes));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});
    CHECK_MALLOC(Mem::malloc((void**)&m3, sizeof(int)));
    utils::Defer free_m3([&m3]() {Mem::free(m3);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(m3, &err_index, sizeof(int), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_apply_dims(size);
    cuda_kernel::one_hot<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, size, m2, classes, m3);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "one_hot");
    CHECK_CUDA_CALL(cudaMemcpy(&err_index, m3, sizeof(int), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    if (err_index >= 0) {
        return err_index;
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * size * classes, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    return err_index;
}

void transpose(f64* ms, int row, int col, int size) {
    int cnt = row * col * size;
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * cnt));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * cnt));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, ms, sizeof(f64) * cnt, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_matrix_dims(row, col, size);
    cuda_kernel::transpose<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, m2, row, col, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "transpose");
    CHECK_CUDA_CALL(cudaMemcpy(ms, m2, sizeof(f64) * cnt, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void mmul(int m, int n, int k, const f64* data1, const f64* data2, f64* dst, int size) {
    cublasHandle_t handle; 
    CHECK_CUBLAS_CALL(cublasCreate(&handle), "create"); 
    utils::Defer destroy_handle([&handle]() {cublasDestroy(handle);});

    f64 *cm, **icm, **hicm;
    CHECK_MALLOC(Mem::malloc((void**)&cm, (m * k + k * n + m * n) * size * sizeof(f64)));
    utils::Defer destroy_cm([&cm]() {Mem::free(cm);});
    CHECK_MALLOC(Mem::malloc((void**)&icm, 3 * size * sizeof(f64*)));
    utils::Defer destroy_icm([&icm]() {Mem::free(icm);});
    hicm = (f64**)malloc(3 * size * sizeof(f64*));
    utils::Defer destroy_hicm([&hicm]() {free(hicm);});

    CHECK_CUDA_CALL(cudaMemcpy(cm, data1, sizeof(f64) * m * k * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(cm + m * k * size, data2, sizeof(f64) * k * n * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    // dst不需要拷贝

    for (int i = 0; i < size; i++) {
        hicm[i] = cm + i * m * k;
        hicm[i + size] = cm + size * m * k + i * k * n;
        hicm[i + 2 * size] = cm + size * m * k + size * k * n + i * m * n;
    }
    CHECK_CUDA_CALL(cudaMemcpy(icm, hicm, 3 * size * sizeof(f64*), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    const f64 alpha = 1.0, beta = 0.0; 
    CHECK_CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, icm + size, n, icm, k, &beta, icm + 2 * size, n, size), "gemm_batched");

    CHECK_CUDA_CALL(cudaMemcpy(dst, cm + m * k * size + k * n * size, sizeof(f64) * m * n * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

bool inv(int m, f64* data, int size) {
    cublasHandle_t handle; 
    CHECK_CUBLAS_CALL(cublasCreate(&handle), "create"); 
    utils::Defer destroy_handle([&handle]() {cublasDestroy(handle);});

    f64 *cm, **icm, **hicm;
    CHECK_MALLOC(Mem::malloc((void**)&cm, 2 * m * m * size * sizeof(f64)));
    utils::Defer destroy_cm([&cm]() {Mem::free(cm);});
    CHECK_MALLOC(Mem::malloc((void**)&icm, 2 * size * sizeof(f64*)));
    utils::Defer destroy_icm([&icm]() {Mem::free(icm);});
    hicm = (f64**)malloc(2 * size * sizeof(f64*));
    utils::Defer destroy_hicm([&hicm]() {free(hicm);});

    CHECK_CUDA_CALL(cudaMemcpy(cm, data, sizeof(f64) * m * m * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    for (int i = 0; i < 2 * size; i++) {
        hicm[i] = cm + i * m * m;
    }
    CHECK_CUDA_CALL(cudaMemcpy(icm, hicm, 2 * size * sizeof(f64*), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    int *pivot_arr, *info_arr, *hinfo_arr;
    CHECK_MALLOC(Mem::malloc((void**)&pivot_arr, m * size * sizeof(int)));
    utils::Defer destroy_pivot_arr([&pivot_arr]() {Mem::free(pivot_arr);});
    CHECK_MALLOC(Mem::malloc((void**)&info_arr, size * sizeof(int)));
    utils::Defer destroy_info_arr([&info_arr]() {Mem::free(info_arr);});
    hinfo_arr = (int*)malloc(size * sizeof(int));
    utils::Defer destroy_hinfo_arr([&hinfo_arr]() {free(hinfo_arr);});

    CHECK_CUBLAS_CALL(cublasDgetrfBatched(handle, m, icm, m, pivot_arr, info_arr, size), "getrf_batched");

    auto check_info = [&hinfo_arr, &info_arr, &size](const std::string& fn) -> bool {
        CHECK_CUDA_CALL(cudaMemcpy(hinfo_arr, info_arr, size * sizeof(int), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
        for (int i = 0; i < size; i++) {
            if (hinfo_arr[i] != 0) {
                LOG(ERROR) << __FUNCTION__ << " check info err: " << fn << " index: " << i << " code: " << hinfo_arr[i];
                return false;
            }
        }
        return true;
    };
    if (!check_info("getrf_batched")) {
        return false;
    }

    CHECK_CUBLAS_CALL(cublasDgetriBatched(handle, m, icm, m, pivot_arr, icm + size, m, info_arr, size), "dgetri_batched");
    if (!check_info("getri_batched")) {
        return false;
    }

    CHECK_CUDA_CALL(cudaMemcpy(data, cm + m * m * size, m * m * size * sizeof(f64), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    return true;
}




}