#ifndef MATRIX_F64_H
#define MATRIX_F64_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/slice.h"
#include <mutex>

namespace cuda {

/*
 * 以矩阵为例（m=2 x n=3）：
 * [1, 2, 3
 *  4, 5, 6]
 * cpu：行存储内存布局 1, 2, 3, 4, 5, 6
 * gpu：列存储内存布局 1, 4, 2, 5, 3, 6
 * cpu -> gpu（同一个矩阵布局）：1. 指定主维数为列数3 2. 转置
 * gpu -> cpu (同一个矩阵布局)：1. 指定主维数为行数2 3. 转置
 */

class MatrixF64 {
public:
  MatrixF64();
  MatrixF64(u32 m, u32 n, const base::Slice<f64> &slice);
  MatrixF64(const MatrixF64&);
  MatrixF64& operator=(const MatrixF64&);

  ~MatrixF64();
  const base::Slice<f64> &getSlice();
  f64 *getCudaSlice();
  std::string toString() const;
  MatrixF64 transpose();
private:
  base::Slice<f64> _slice;
  u32 _m = 0, _n = 0;
  std::mutex _cuda_slice_lock;
  f64 *_cuda_slice = nullptr;
};

}

#endif