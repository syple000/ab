#include "auto_engine/base/exit_code.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/cuda/mem.h"
#include "auto_engine/cuda/tensor.h"
#include "auto_engine/cuda/kernel.cuh"
#include "auto_engine/shape/shape.h"
#include "auto_engine/utils/defer.h"
#include "cublas_v2.h"
#include <cstdlib>
#include <cstring>
#include <fmt/core.h>
#include <iostream>
#include <tuple>
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
    auto gridx_dim = (size + tcnt_per_block() - 1) / tcnt_per_block();
    cuda_kernel::sum<<<dim3(gridx_dim), dim3(tcnt_per_block())>>>(m1, size, m2);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand(const f64* src, f64* dst, u32 size) {
    f64 *m1, *m2;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64)));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * size));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto gridx_dim = (size + tcnt_per_block() - 1) / tcnt_per_block();
    cuda_kernel::expand<<<dim3(gridx_dim), dim3(tcnt_per_block())>>>(m1, m2, size);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand");
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * size, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void one_hot(const f64* src, u32 size, f64* dst, u32 classes, u32* err_occur, u32* err_index) {
    f64 *m1, *m2; u32 *m3;
    CHECK_MALLOC(Mem::malloc((void**)&m1, sizeof(f64) * size));
    utils::Defer free_m1([&m1]() {Mem::free(m1);});
    CHECK_MALLOC(Mem::malloc((void**)&m2, sizeof(f64) * size * classes));
    utils::Defer free_m2([&m2]() {Mem::free(m2);});
    CHECK_MALLOC(Mem::malloc((void**)&m3, sizeof(u32) * 2));
    utils::Defer free_m3([&m3]() {Mem::free(m3);});

    CHECK_CUDA_CALL(cudaMemcpy(m1, src, sizeof(f64) * size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(m3, err_occur, sizeof(u32), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    auto dims = get_apply_dims(size * classes);
    cuda_kernel::one_hot<<<std::get<0>(dims), std::get<1>(dims)>>>(m1, size, m2, classes, m3, m3 + 1);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "one_hot");
    CHECK_CUDA_CALL(cudaMemcpy(err_occur, m3, sizeof(u32), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    if (*err_occur == 1) {
        CHECK_CUDA_CALL(cudaMemcpy(err_index, m3 + 1, sizeof(u32), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
        return;
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, m2, sizeof(f64) * size * classes, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void sum(const f64* src, const base::Shape& shape, f64* dst, u32 d) {
    f64 *msrc, *mdst;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * shape.tensorSize() / shape.getDim(d)));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemset(mdst, 0, sizeof(f64) * shape.tensorSize() / shape.getDim(d)), "cuda_memset");

    if (d == shape.dimCnt() - 1) {
        u32 gridy_dim = (shape.tensorSize() / shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = (shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::sum_l<<<dim3(gridx_dim, gridy_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, shape.tensorSize(), shape.getDim(shape.dimCnt() - 1));
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum_l");
    } else {
        u32 gridy_dim = (shape.getStrides()[d] + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridz_dim = (shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = shape.tensorSize() / shape.getStrides()[d] / shape.getDim(d);
        cuda_kernel::sum_nl<<<dim3(gridx_dim, gridy_dim, gridz_dim), dim3(1, sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, shape.tensorSize(), shape.getDim(d), shape.getStrides()[d]);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "sum_nl");
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * shape.tensorSize() / shape.getDim(d), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void expand(const f64* src, f64* dst, const base::Shape& shape, u32 d) {
    f64 *msrc, *mdst;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * shape.tensorSize() / shape.getDim(d)));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * shape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * shape.tensorSize() / shape.getDim(d), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    if (d == shape.dimCnt() - 1) {
        u32 gridy_dim = (shape.tensorSize() / shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = (shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::expand_l<<<dim3(gridx_dim, gridy_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, shape.tensorSize(), shape.getDim(d));
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand_l");
    } else {
        u32 gridy_dim = (shape.getStrides()[d] + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridz_dim = (shape.getDim(d) + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        u32 gridx_dim = shape.tensorSize() / shape.getStrides()[d] / shape.getDim(d);
        cuda_kernel::expand_nl<<<dim3(gridx_dim, gridy_dim, gridz_dim), dim3(1, sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, shape.tensorSize(), shape.getDim(d), shape.getStrides()[d]);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "expand_nl");
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void cat(const f64* src, const base::Shape& src_shape, f64* dst, const base::Shape& dst_shape, u32 d, u32 d_offset) {
    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * src_shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * dst_shape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * src_shape.dimCnt() * 4));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * src_shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mdst, dst, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, src_shape.getDims().data(), sizeof(u32) * src_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + src_shape.dimCnt(), src_shape.getStrides().data(), sizeof(u32) * src_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 2* dst_shape.dimCnt(), dst_shape.getDims().data(), sizeof(u32) * dst_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 3 * dst_shape.dimCnt(), dst_shape.getStrides().data(), sizeof(u32) * dst_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto gridx_dim = (src_shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::cat<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(msrc, mds, mds + src_shape.dimCnt(), mdst, mds + 2 * dst_shape.dimCnt(), mds + 3 * dst_shape.dimCnt(), src_shape.dimCnt(), d, d_offset);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "cat");
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void cat(const std::vector<const f64*>& srcs, const std::vector<std::reference_wrapper<const base::Shape>>& srcs_shapes, f64* dst, const base::Shape& dst_shape, u32 d) {
    // 先全部赋值到内存中，后进行一次拷贝，最后进行index梳理
    u32 total_size = 0;
    for (u32 i = 0; i < srcs_shapes.size(); i++) {
        const auto& s = srcs_shapes[i].get();
        total_size += s.tensorSize() * sizeof(f64);
        total_size += 2 * s.dimCnt() * sizeof(u32);
    }
    total_size += dst_shape.tensorSize() * sizeof(f64);
    total_size += 2 * dst_shape.dimCnt() * sizeof(u32);

    auto lm = (void*)malloc(total_size);
    utils::Defer free_lm([&lm] {free(lm);});
    auto rlm = lm;
    for (u32 i = 0; i < srcs_shapes.size(); i++) {
        const auto& shape = srcs_shapes[i].get();
        memcpy(rlm, srcs[i], shape.tensorSize() * sizeof(f64));
        rlm = ((f64*)rlm) + shape.tensorSize();
        memcpy(rlm, shape.getDims().data(), shape.dimCnt() * sizeof(u32));
        rlm = ((u32*)rlm) + shape.dimCnt();
        memcpy(rlm, shape.getStrides().data(), shape.dimCnt() * sizeof(u32));
        rlm = ((u32*)rlm) + shape.dimCnt();
    }
    rlm = ((f64*)rlm) + dst_shape.tensorSize();
    memcpy(rlm, dst_shape.getDims().data(), dst_shape.dimCnt() * sizeof(u32));
    rlm = ((u32*)rlm) + dst_shape.dimCnt();
    memcpy(rlm, dst_shape.getStrides().data(), dst_shape.dimCnt() * sizeof(u32));

    void* m;
    CHECK_MALLOC(Mem::malloc(&m, total_size));
    utils::Defer free_m([&m] {Mem::free(m);});
    CHECK_CUDA_CALL(cudaMemcpy(m, lm, total_size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto lm_index = (void**)malloc(srcs_shapes.size() * sizeof(void*) * 3);
    utils::Defer free_lm_index([&lm_index] {free(lm_index);});
    auto rm = m;
    for (u32 i = 0; i < srcs_shapes.size(); i++) {
        const auto& shape = srcs_shapes[i].get();
        lm_index[i] = rm;
        rm = ((f64*)rm) + shape.tensorSize();
        lm_index[i + srcs_shapes.size()] = rm;
        rm = ((u32*)rm) + shape.dimCnt();
        lm_index[i + srcs_shapes.size() * 2] = rm;
        rm = ((u32*)rm) + shape.dimCnt();
    }
    void* m_index;
    CHECK_MALLOC(Mem::malloc(&m_index, srcs_shapes.size() * sizeof(void*) * 3));
    utils::Defer free_m_index([&m_index] {Mem::free(m_index);});
    CHECK_CUDA_CALL(cudaMemcpy(m_index, lm_index, srcs_shapes.size() * sizeof(void*) * 3, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto dst_m = (f64*)rm;
    rm = ((f64*)rm) + dst_shape.tensorSize();
    auto dst_dims_m = (const u32*)rm;
    rm = ((u32*)rm) + dst_shape.dimCnt();
    auto dst_strides_m = (const u32*)rm;

    auto gridx_dim = (dst_shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::cat<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>((const f64**)m_index, (const u32**)m_index + srcs_shapes.size(), (const u32**)m_index + 2 * srcs_shapes.size(), srcs_shapes.size(), dst_m, dst_dims_m, dst_strides_m, dst_shape.dimCnt(), d);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "cat");
    CHECK_CUDA_CALL(cudaMemcpy(dst, dst_m, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void split(const f64* src, const base::Shape& src_shape, f64* dst, const base::Shape& dst_shape, u32 d, u32 d_offset) {
    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * src_shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * dst_shape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * src_shape.dimCnt() * 4));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * src_shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mdst, dst, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, src_shape.getDims().data(), sizeof(u32) * src_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + src_shape.dimCnt(), src_shape.getStrides().data(), sizeof(u32) * src_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 2* dst_shape.dimCnt(), dst_shape.getDims().data(), sizeof(u32) * dst_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + 3 * dst_shape.dimCnt(), dst_shape.getStrides().data(), sizeof(u32) * dst_shape.dimCnt(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto gridx_dim = (dst_shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::split<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(msrc, mds, mds + src_shape.dimCnt(), mdst, mds + 2 * dst_shape.dimCnt(), mds + 3 * dst_shape.dimCnt(), src_shape.dimCnt(), d, d_offset);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "split");
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
}

void split(const f64* src, const base::Shape& src_shape, const std::vector<f64*>& dst, const std::vector<std::reference_wrapper<const base::Shape>>& dsts_shapes, u32 d) {
    // 先全部赋值到内存中，后进行一次拷贝，最后进行index梳理
    u32 total_size = 0;
    for (u32 i = 0; i < dsts_shapes.size(); i++) {
        const auto& s = dsts_shapes[i].get();
        total_size += s.tensorSize() * sizeof(f64);
        total_size += 2 * s.dimCnt() * sizeof(u32);
    }
    total_size += src_shape.tensorSize() * sizeof(f64);
    total_size += 2 * src_shape.dimCnt() * sizeof(u32);

    auto lm = (void*)malloc(total_size);
    utils::Defer free_lm([&lm] {free(lm);});
    auto rlm = lm;
    for (u32 i = 0; i < dsts_shapes.size(); i++) {
        const auto& shape = dsts_shapes[i].get();
        // memcpy(rlm, dst[i], shape.tensorSize() * sizeof(f64));
        rlm = ((f64*)rlm) + shape.tensorSize();
        memcpy(rlm, shape.getDims().data(), shape.dimCnt() * sizeof(u32));
        rlm = ((u32*)rlm) + shape.dimCnt();
        memcpy(rlm, shape.getStrides().data(), shape.dimCnt() * sizeof(u32));
        rlm = ((u32*)rlm) + shape.dimCnt();
    }
    memcpy(rlm, src, src_shape.tensorSize() * sizeof(f64));
    rlm = ((f64*)rlm) + src_shape.tensorSize();
    memcpy(rlm, src_shape.getDims().data(), src_shape.dimCnt() * sizeof(u32));
    rlm = ((u32*)rlm) + src_shape.dimCnt();
    memcpy(rlm, src_shape.getStrides().data(), src_shape.dimCnt() * sizeof(u32));

    void* m;
    CHECK_MALLOC(Mem::malloc(&m, total_size));
    utils::Defer free_m([&m] {Mem::free(m);});
    CHECK_CUDA_CALL(cudaMemcpy(m, lm, total_size, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto lm_index = (void**)malloc(dsts_shapes.size() * sizeof(void*) * 3);
    utils::Defer free_lm_index([&lm_index] {free(lm_index);});
    auto rm = m;
    for (u32 i = 0; i < dsts_shapes.size(); i++) {
        const auto& shape = dsts_shapes[i].get();
        lm_index[i] = rm;
        rm = ((f64*)rm) + shape.tensorSize();
        lm_index[i + dsts_shapes.size()] = rm;
        rm = ((u32*)rm) + shape.dimCnt();
        lm_index[i + dsts_shapes.size() * 2] = rm;
        rm = ((u32*)rm) + shape.dimCnt();
    }
    void* m_index;
    CHECK_MALLOC(Mem::malloc(&m_index, dsts_shapes.size() * sizeof(void*) * 3));
    utils::Defer free_m_index([&m_index] {Mem::free(m_index);});
    CHECK_CUDA_CALL(cudaMemcpy(m_index, lm_index, dsts_shapes.size() * sizeof(void*) * 3, cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    auto src_m = (const f64*)rm;
    rm = ((f64*)rm) + src_shape.tensorSize();
    auto src_dims_m = (const u32*)rm;
    rm = ((u32*)rm) + src_shape.dimCnt();
    auto src_strides_m = (const u32*)rm;

    auto gridx_dim = (src_shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
    cuda_kernel::split<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(src_m, src_dims_m, src_strides_m, (f64**)m_index, (const u32**)m_index + dsts_shapes.size(), (const u32**)m_index + 2 * dsts_shapes.size(), dsts_shapes.size(), src_shape.dimCnt(), d);
    CHECK_CUDA_CALL(cudaPeekAtLastError(), "split");
    CHECK_CUDA_CALL(cudaMemcpy(lm, m, total_size - src_shape.tensorSize() * sizeof(f64) - src_shape.dimCnt() * sizeof(u32) * 2, cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
    rlm = lm;
    for (u32 i = 0; i < dsts_shapes.size(); i++) {
        const auto& shape = dsts_shapes[i].get();
        memcpy(dst[i], rlm, shape.tensorSize() * sizeof(f64));
        rlm = ((f64*)rlm) + shape.tensorSize();
        rlm = ((u32*)rlm) + shape.dimCnt() * 2;
    }
}

void permute(const f64* src, const base::Shape& src_shape, f64* dst, const base::Shape& dst_shape, const std::vector<u32>& pl) {
    f64 *msrc, *mdst;
    u32 *mds;
    CHECK_MALLOC(Mem::malloc((void**)&msrc, sizeof(f64) * src_shape.tensorSize()));
    utils::Defer free_msrc([&msrc]() {Mem::free(msrc);});
    CHECK_MALLOC(Mem::malloc((void**)&mdst, sizeof(f64) * dst_shape.tensorSize()));
    utils::Defer free_mdst([&mdst]() {Mem::free(mdst);});
    CHECK_MALLOC(Mem::malloc((void**)&mds, sizeof(u32) * pl.size() * 3));
    utils::Defer free_mds([&mds]() {Mem::free(mds);});

    CHECK_CUDA_CALL(cudaMemcpy(msrc, src, sizeof(f64) * src_shape.tensorSize(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds, src_shape.getStrides().data(), sizeof(u32) * pl.size(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + pl.size(), dst_shape.getStrides().data(), sizeof(u32) * pl.size(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");
    CHECK_CUDA_CALL(cudaMemcpy(mds + pl.size() * 2, pl.data(), sizeof(u32) * pl.size(), cudaMemcpyHostToDevice), "cuda_memcpy_h2d");

    int pl_index = pl.size() - 1;
    int index = src_shape.dimCnt() - 1;
    while (pl_index >= 0 && src_shape.getDim(pl[pl_index]) == 1) {pl_index -= 1;} 
    while (index >= 0 && src_shape.getDim(index) == 1) {index -= 1;}

    if (pl_index >= 0 && index >= 0 && pl[pl_index] != index) {
        auto stride = src_shape.getStrides()[pl[pl_index]];
        u32 gridx_dim = (stride + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();

        u32 gridz_dim = (src_shape.tensorSize() / stride + grid_cnt_y() * sqrt_tcnt_per_block() - 1) / (grid_cnt_y() * sqrt_tcnt_per_block());
        u32 gridy_dim;
        if (gridz_dim > 1) {
            gridy_dim = grid_cnt_y();
        } else {
            gridy_dim = (src_shape.tensorSize() / stride + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        }
        cuda_kernel::permute_l<<<dim3(gridx_dim, gridy_dim, gridz_dim), dim3(sqrt_tcnt_per_block(), sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + pl.size(), src_shape.dimCnt(), src_shape.tensorSize(), mds + 2 * pl.size(), pl_index);
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "permute_l");
    } else {
        u32 gridx_dim = (src_shape.tensorSize() + sqrt_tcnt_per_block() - 1) / sqrt_tcnt_per_block();
        cuda_kernel::permute_nl<<<dim3(gridx_dim), dim3(sqrt_tcnt_per_block())>>>(msrc, mdst, mds, mds + pl.size(), src_shape.dimCnt(), src_shape.tensorSize(), mds + 2 * pl.size());
        CHECK_CUDA_CALL(cudaPeekAtLastError(), "permute_nl");
    }
    CHECK_CUDA_CALL(cudaMemcpy(dst, mdst, sizeof(f64) * dst_shape.tensorSize(), cudaMemcpyDeviceToHost), "cuda_memcpy_d2h");
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