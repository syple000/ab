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

MatrixF64::MatrixF64() {} 
MatrixF64::MatrixF64(u32 m, u32 n, const base::Slice<f64>& slice): _m(m), _n(n), _slice(slice) { 
    if (m * n != slice.size()) { 
        LOG(ERROR) << "matrix size invalid"; 
        _m = 0; 
        _n = 0; 
        _slice = base::Slice<f64>(); 
        return; 
    } 
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
MatrixF64::~MatrixF64() { 
    auto guard = std::lock_guard<std::mutex>(_cuda_slice_lock); 
    if (_cuda_slice) { 
        Mem::free((void*)_cuda_slice); 
    } 
} 
const base::Slice<f64>& MatrixF64::getSlice() {
    return _slice;
} 
f64* MatrixF64::getCudaSlice() { 
    if (_cuda_slice) { 
        return _cuda_slice; 
    } 
    auto guard = std::lock_guard<std::mutex>(_cuda_slice_lock); 
    if (_cuda_slice) { 
        return _cuda_slice; 
    } 
    _cuda_slice = reinterpret_cast<f64*>(Mem::host2Device(_slice.data(), _slice.size() * sizeof(f64))); 
    return _cuda_slice; 
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

MatrixF64 MatrixF64::transpose() { 
    cublasHandle_t handle; 
    cublasCreate(&handle); 
    utils::Defer destroyHandle([&handle]{cublasDestroy(handle);});
    const f64 alpha = 1.0, beta = 0.0; 
    auto cuda_ret = (f64*)Mem::malloc(_slice.size() * sizeof(f64)); 
    if (!cuda_ret) { 
        LOG(ERROR) << "cuda mem alloc err"; 
        return MatrixF64(); 
    } 
    utils::Defer freeMem([&cuda_ret] {Mem::free(cuda_ret);});
    auto status = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, _m, _n, &alpha, reinterpret_cast<const f64*>(getCudaSlice()), _n,  
        &beta, reinterpret_cast<const f64*>(getCudaSlice()), _m, cuda_ret, _m); 
    if (status != CUBLAS_STATUS_SUCCESS) { 
        LOG(ERROR) << "cublas geam err: " << status; 
        return MatrixF64(); 
    } 
    auto ret = Mem::device2Host(cuda_ret, _slice.size() * sizeof(f64)); 
    return MatrixF64(_n, _m, base::Slice<f64>(std::make_shared<std::vector<f64>>(reinterpret_cast<f64*>(ret), reinterpret_cast<f64*>(ret)+_slice.size())));  
}

}