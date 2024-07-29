#ifndef CUDA_TENSOR_H
#define CUDA_TENSOR_H

#include "auto_engine/base/basic_types.h"

namespace cuda {

void apply_sin(f64* data, int size);
void apply_cos(f64* data, int size);
void apply_log(f64* data, int size);
void apply_neg(f64* data, int size);
void apply_sign(f64* data, int size);
void apply_abs(f64* data, int size);
void apply_add(f64* data1, const f64* data2, int size);
void apply_sub(f64* data1, const f64* data2, int size);
void apply_mul(f64* data1, const f64* data2, int size);
void apply_div(f64* data1, const f64* data2, int size);
void apply_pow(f64* data1, const f64* data2, int size);
void apply_add(f64* data, f64 n, int size);
void apply_sub(f64* data, f64 n, int size);
void apply_mul(f64* data, f64 n, int size);
void apply_div(f64* data, f64 n, int size);
void apply_pow(f64* data, f64 n, int size);
void apply_sum(const f64* src, int size, f64* dst);
void apply_expand(const f64* src, f64* dst, int size);
int ont_hot(const f64* src, int size, f64* dst, int classes);
void sum_along_row(const f64* srcs, f64* dsts, int row, int col, int size);
void expand_along_row(const f64* src, f64* dst, int row, int col, int size);
void sum_along_col(const f64* srcs, f64* dsts, int row, int col, int size);
void expand_along_col(const f64* src, f64* dst, int row, int col, int size);
void transpose(f64* ms, int row, int col, int size);
void mmul(int m, int n, int k, const f64* data1, const f64* data2, f64* dst, int size);
bool inv(int m, f64* data, int size);

}

#endif