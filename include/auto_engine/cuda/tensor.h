#ifndef CUDA_TENSOR_H
#define CUDA_TENSOR_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/shape/shape.h"

namespace cuda {

void apply_sin(f64* data, u32 size);
void apply_cos(f64* data, u32 size);
void apply_log(f64* data, u32 size);
void apply_neg(f64* data, u32 size);
void apply_sign(f64* data, u32 size);
void apply_abs(f64* data, u32 size);
void apply_add(f64* data1, const f64* data2, u32 size);
void apply_sub(f64* data1, const f64* data2, u32 size);
void apply_mul(f64* data1, const f64* data2, u32 size);
void apply_div(f64* data1, const f64* data2, u32 size);
void apply_pow(f64* data1, const f64* data2, u32 size);
void apply_add(f64* data, f64 n, u32 size);
void apply_sub(f64* data, f64 n, u32 size);
void apply_mul(f64* data, f64 n, u32 size);
void apply_div(f64* data, f64 n, u32 size);
void apply_pow(f64* data, f64 n, u32 size);
void sum(const f64* src, u32 size, f64* dst);
void expand(const f64* src, f64* dst, u32 size);
void sum(const f64* src, f64* dst, const base::Shape&, u32 d);
void expand(const f64* src, f64* dst, const base::Shape&, u32 d, u32 expd);
void cat(const f64* src1, const base::Shape&, const f64* src2, const base::Shape&, f64* dst, const base::Shape&, u32 d);
void split(const f64* src, const base::Shape&, f64* dst1, const base::Shape&, f64* dst2, const base::Shape&, u32 d);
void transpose(f64* data, const base::Shape&,  const base::Shape&, u32 d1, u32 d2);
void mmul(u32 m, u32 n, u32 k, const f64* data1, const f64* data2, f64* dst, u32 size);
bool inv(u32 m, f64* data, u32 size);
int ont_hot(const f64* src, u32 size, f64* dst, u32 classes);

}

#endif