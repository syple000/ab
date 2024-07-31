#include "auto_engine/base/exit_code.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/cuda/mem.h"
#include "auto_engine/cuda/tensor.h"
#include "auto_engine/cuda/kernel.cuh"
#include "auto_engine/shape/shape.h"
#include "auto_engine/utils/defer.h"
#include "auto_engine/utils/maths.h"
#include "cublas_v2.h"
#include <algorithm>
#include <cstdlib>
#include <fmt/core.h>
#include <tuple>
#include <utility>
#include <vector>

namespace cuda {

#define CHECK_MALLOC(call) \
{ \
    auto succ = call; \
    if (!succ) { \
        LOG(ERROR) << __FUNCTION__ << " malloc cuda err"; \
        exit(CUDA_ERR); \
    } \
} \

std::tuple<dim3, dim3> get_apply_dims(u32 size) {
    u32 grid_dim = (size + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    return std::tuple(dim3(grid_dim), dim3(sqrt_tcnt_per_block())); 
}

#define DEFINE_APPLY_1E(fn, T) \
void fn(T* data, u32 size) { \
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
void fn(T* data1, const T* data2, u32 size) { \
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
void fn(T* data, T n, u32 size) { \
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

void sum(const f64* src, u32 size, f64* dst) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64)));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(m2, 0, sizeof(f64)), "cuda_memset");
    auto blockx_dim = utils::nextPowerOfTwo(std::min(tcnt_per_block(), size));
    auto gridx_dim = (size + blockx_dim - 1) / blockx_dim;
    cuda_kernel::sum<<<dim3(gridx_dim), dim3(blockx_dim)>>>(m1, size, m2);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply_sum");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand(const f64* src, f64* dst, u32 size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64)));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto blockx_dim = utils::nextPowerOfTwo(std::min(tcnt_per_block(), size));
    auto gridx_dim = (size + blockx_dim - 1) / blockx_dim;
    cuda_kernel::expand<<<dim3(gridx_dim), dim3(blockx_dim)>>>(m1, m2, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "apply_expand");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

int ont_hot(const f64* src, u32 size, f64* dst, u32 classes) {
    int err_index = -1;

    f64 *m1, *m2; u32 *m3;
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


void sum(const f64* src, f64* dst, const base::Shape& shape, u32 d) {
    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * shape.tensorSize() / shape.getDim(d)));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * shape.dimCnt() * 2));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, shape.getDims().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + shape.dimCnt(), shape.getStrides().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(mdst, 0, sizeof(f64) * shape.tensorSize() / shape.getDim(d)), "cuda_memset");

    if (d == shape.dimCnt() - 1) {
        u32 gridy_dim = (shape.tensorSize() / shape.getDim(shape.dimCnt() - 1) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = (shape.getDim(shape.dimCnt() - 1) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::sum<<<dim3(gridx_dim, gridy_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), shape.dimCnt());
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum1");
    } else {
        u32 gridy_dim = (shape.getStrides()[d] + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridz_dim = (shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = shape.tensorSize() / shape.getStrides()[d] / shape.getDim(d);
        cuda_kernel::sum<<<dim3(gridx_dim, gridy_dim, gridz_dim), dim3(1, sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), shape.dimCnt(), d);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum2");
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * shape.tensorSize() / shape.getDim(d), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand(const f64* src, f64* dst, const base::Shape& shape, u32 d, u32 expd) {
    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * shape.tensorSize() * expd));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * shape.dimCnt() * 2));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, shape.getDims().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + shape.dimCnt(), shape.getStrides().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(mdst, 0, sizeof(f64) * shape.tensorSize() * expd), "cuda_memset");

    if (d == shape.dimCnt()) {
        u32 gridy_dim = (shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = (expd + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::expand<<<dim3(gridx_dim, gridy_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), shape.dimCnt(), expd);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand1");
    } else {
        u32 gridy_dim = (shape.getStrides()[d] * shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridz_dim = (expd + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = shape.tensorSize() / shape.getStrides()[d] / shape.getDim(d);
        cuda_kernel::expand<<<dim3(gridx_dim, gridy_dim, gridz_dim), dim3(1, sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), shape.dimCnt(), d, expd);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand2");
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * shape.tensorSize() * expd, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}


void cat(const f64* src1, const base::Shape& shape1, const f64* src2, const base::Shape& shape2, f64* dst, const base::Shape& oshape, u32 d) {
    f64 *msrc1, *msrc2, *mdst;
    u32 *mds;

    CHECK_MALLOC(Mem::malloc((void**)&msrc1, sizeof(f64) * shape1.tensorSize()));
    utils::Defer free_msrc1([&msrc1]() {Mem::free(msrc1);});
    CHECK_MALLOC(Mem::malloc((void**)&msrc2, sizeof(f64) * shape2.tensorSize()));
    utils::Defer free_msrc2([&msrc2]() {Mem::free(msrc2);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * oshape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * oshape.dimCnt() * 6));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc1, src1, sizeof(f64) * shape1.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(msrc2, src2, sizeof(f64) * shape2.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, shape1.getDims().data(), sizeof(u32) * shape1.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + shape1.dimCnt(), shape1.getStrides().data(), sizeof(u32) * shape1.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 2 * shape2.dimCnt(), shape2.getDims().data(), sizeof(u32) * shape2.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 3 * shape2.dimCnt(), shape2.getStrides().data(), sizeof(u32) * shape2.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 4 * oshape.dimCnt(), oshape.getDims().data(), sizeof(u32) * oshape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 5 * oshape.dimCnt(), oshape.getStrides().data(), sizeof(u32) * oshape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto gridx_dim = (oshape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::cat<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(msrc1, mds, mds + shape1.dimCnt(), msrc2, mds + 2 * shape2.dimCnt(), mds + 3 * shape2.dimCnt(), mdst, mds + 4 * oshape.dimCnt(), mds + 5 * oshape.dimCnt(), oshape.dimCnt(), d);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "cat");
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * oshape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}


void split(const f64* src, const base::Shape& ishape, f64* dst1, const base::Shape& shape1, f64* dst2, const base::Shape& shape2, u32 d) {
    f64 *msrc, *mdst1, *mdst2;
    u32 *mds;

    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * ishape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst1, sizeof(f64) * shape1.tensorSize()));
    utils::Defer free_mdst1([&mdst1]() {Mem::free(mdst1);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst2, sizeof(f64) * shape2.tensorSize()));
    utils::Defer free_mdst([&mdst2]() {Mem::free(mdst2);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * ishape.dimCnt() * 6));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * ishape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, ishape.getDims().data(), sizeof(u32) * ishape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + ishape.dimCnt(), ishape.getStrides().data(), sizeof(u32) * ishape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 2 * shape1.dimCnt(), shape1.getDims().data(), sizeof(u32) * shape1.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 3 * shape1.dimCnt(), shape1.getStrides().data(), sizeof(u32) * shape1.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 4 * shape2.dimCnt(), shape2.getDims().data(), sizeof(u32) * shape2.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 5 * shape2.dimCnt(), shape2.getStrides().data(), sizeof(u32) * shape2.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto gridx_dim = (ishape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::split<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(msrc, mds, mds + ishape.dimCnt(), mdst1, mds + 2 * shape1.dimCnt(), mds + 3 * shape1.dimCnt(), mdst2, mds + 4 * shape2.dimCnt(), mds + 5 * shape2.dimCnt(), ishape.dimCnt(), d);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "cat");
    CHECK_CUDA_CALL(cudaMemcpy(dst1, mdst1, sizeof(f64) * shape1.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    CHECK_CUDA_CALL(cudaMemcpy(dst2, mdst2, sizeof(f64) * shape2.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void transpose(f64* data, const base::Shape& shape, const base::Shape& transpose_shape, u32 d1, u32 d2) {
    if (d1 > d2) {std::swap(d1, d2);}

    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * transpose_shape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * shape.dimCnt() * 3));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, data, sizeof(f64) * shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, shape.getDims().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + shape.dimCnt(), shape.getStrides().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + shape.dimCnt() * 2, transpose_shape.getStrides().data(), sizeof(u32) * shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    if (d2 == shape.dimCnt() - 1) {
        u32 gridy_dim = (shape.tensorSize() / shape.getStrides()[d1] + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = (shape.getStrides()[d1] + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::transpose<<<dim3(gridx_dim, gridy_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), mds + shape.dimCnt() * 2, shape.dimCnt(), d1);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "tranpose1");
    } else {
        u32 grid_dim = (shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::transpose<<<dim3(grid_dim), dim3(sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + shape.dimCnt(), mds + shape.dimCnt() * 2, shape.dimCnt(), d1, d2);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "tranpose2");
    }

    CHECK_CUDA_CALL(cudaMemcpy(data, mdst, sizeof(f64) * shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void mmul(u32 m, u32 n, u32 k, const f64* data1, const f64* data2, f64* dst, u32 size) {
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

    for (u32 i = 0; i < size; i++) {
        hicm[i] = cm + i * m * k;
        hicm[i + size] = cm + size * m * k + i * k * n;
        hicm[i + 2 * size] = cm + size * m * k + size * k * n + i * m * n;
    }
    CHECK_CUDA_CALL(cudaMemcpy(icm, hicm, 3 * size * sizeof(f64*), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    const f64 alpha = 1.0, beta = 0.0; 
    CHECK_CUBLAS_CALL(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, icm + size, n, icm, k, &beta, icm + 2 * size, n, size), "gemm_batched");

    CHECK_CUDA_CALL(cudaMemcpy(dst, cm + m * k * size + k * n * size, sizeof(f64) * m * n * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

bool inv(u32 m, f64* data, u32 size) {
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
    for (u32 i = 0; i < 2 * size; i++) {
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
        for (u32 i = 0; i < size; i++) {
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