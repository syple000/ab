#ifndef BASE_TENSOR_H 
#define BASE_TENSOR_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/slice.h"
#include "auto_engine/config/config.h"
#include "auto_engine/cuda/matrix_f64.h"
#include "shape.h"
#include "glog/logging.h"
#include "Eigen/Dense"
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace base {

template<typename T>
struct Tensor {
public:
    static_assert(std::is_arithmetic<T>::value, "T should be number");
    Tensor() {}
    Tensor(const Shape&);
    Tensor(const Shape&, const std::function<T()>&);
    Tensor(const Shape&, const T&);
    Tensor(const Shape&, const std::vector<T>&);
    Tensor(const Shape&, std::vector<T>&&);

    Tensor(const Tensor<T>&);
    Tensor(Tensor<T>&&);
    Tensor<T>& operator=(const Tensor<T>&);
    Tensor<T>& operator=(Tensor<T>&&);
    bool operator==(const Tensor<T>&) const;

    Tensor<T> operator+(const Tensor<T>&) const;
    Tensor<T> operator-(const Tensor<T>&) const;
    Tensor<T> operator-() const;
    Tensor<T> log() const;
    Tensor<T> sin() const;
    Tensor<T> cos() const;
    Tensor<T> operator*(const Tensor<T>&) const;
    Tensor<T> operator/(const Tensor<T>&) const;
    Tensor<T> pow(const Tensor<T>&) const;

    // 对齐pytorch语义
    Tensor<T> transpose() const;
    Tensor<T> mmul(const Tensor<T>&) const;
    Tensor<T> inv() const;

    const T& operator()(const std::vector<u32>& index) const;
    T& operator()(const std::vector<u32>& index);
    // 区间生效范围：[left, right]
    bool assign(const std::vector<u32>& start_index, const std::vector<u32>& end_index, const T& e);

    const Shape& shape() const {return _shape;}
    Tensor<T> reshape(const std::vector<u32>& dims) const;
    bool reshapeInplace(const std::vector<u32>& dims);

    std::string toString(bool compact = false) const;
    const std::vector<T>& getData() const;

private:
    Shape _shape;
    std::vector<T> _data;

    bool checkIndex(const std::vector<u32>& index) const;
    u32 calcOffset(const std::vector<u32>& index) const;

    static T _INVALID_NUM;
};

template<typename T>
T Tensor<T>::_INVALID_NUM = std::numeric_limits<T>::quiet_NaN();


template<typename T>
Tensor<T>::Tensor(const Shape& shape) {
    _shape = shape;
    _data = std::vector<T>(shape.tensorSize());
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, const std::function<T()>& f) {
    _shape = shape;
    _data = std::vector<T>(shape.tensorSize());
    for (auto& t : _data) {
        t = f();
    }
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, const T& elem) {
    _shape = shape;
    _data = std::vector<T>(shape.tensorSize(), elem);
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, const std::vector<T>& data) {
    if (shape.tensorSize() != data.size()) {
        LOG(ERROR) << "shape count != data size";
        return;
    }
    _shape = shape;
    _data = data;
}

template<typename T>
Tensor<T>::Tensor(const Shape& shape, std::vector<T>&& data) {
    if (shape.tensorSize() != data.size()) {
        LOG(ERROR) << "shape count: " << shape.tensorSize() << " != data size: " << data.size();
        return;
    }
    _shape = shape;
    _data = std::move(data);
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& t) {
    _shape = t._shape;
    _data = t._data;
}

template<typename T>
Tensor<T>::Tensor(Tensor<T>&& t) {
    _shape = std::move(t._shape);
    _data = std::move(t._data);
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& t) {
    if (this == &t) {
        return *this;
    }
    _shape = t._shape;
    _data = t._data;
    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& t) {
    if (this == &t) {
        return *this;
    }
    _shape = std::move(t._shape);
    _data = std::move(t._data);

    return *this;
}

template<typename T>
bool Tensor<T>::operator==(const Tensor<T>& t) const {
    if (!this) {
        return false;
    }
    if (this == &t) {
        return true;
    }
    if (!(_shape == t._shape)) {
        return false;
    }
    for (int i = 0; i < _data.size(); i++) {
        if (std::isnan(_data[i]) || std::isnan(t._data[i])) {
            if (!std::isnan(_data[i]) || !std::isnan(t._data[i])) {
                return false;
            }
        }
        if (std::abs(_data[i] - t._data[i]) >= EPSILON) {
            return false;
        }
    } 
    return true;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (!(_shape == t._shape)) {
        LOG(ERROR) << "add diff shape tensor";
        return {};
    }

    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        cuda::MatrixF64 m(t._data.size(), 1, base::Slice<T>(&t._data));
        auto rm = sm.add(m);
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            res._data[i] += t._data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (!(_shape == t._shape)) {
        LOG(ERROR) << "mul diff shape tensor";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        cuda::MatrixF64 m(t._data.size(), 1, base::Slice<T>(&t._data));
        auto rm = sm.mul(m);
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            res._data[i] *= t._data[i];
        }
    }
    return std::move(res);
}


template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (!(_shape == t._shape)) {
        LOG(ERROR) << "sub diff shape tensor";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        cuda::MatrixF64 m(t._data.size(), 1, base::Slice<T>(&t._data));
        auto rm = sm.sub(m);
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            res._data[i] -= t._data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::log() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        auto rm = sm.log();
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[i] = ::log(_data[i]);
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::cos() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        auto rm = sm.cos();
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[i] = ::cos(_data[i]);
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::sin() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        auto rm = sm.sin();
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[i] = ::sin(_data[i]);
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::operator-() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        auto rm = sm.neg();
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[i] = -_data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (!(_shape == t._shape)) {
        LOG(ERROR) << "div diff shape tensor";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        cuda::MatrixF64 m(t._data.size(), 1, base::Slice<T>(&t._data));
        auto rm = sm.div(m);
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            res._data[i] /= t._data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (!(_shape == t._shape)) {
        LOG(ERROR) << "div diff shape tensor";
        return {};
    }
    Tensor<T> res(_shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::MatrixF64 sm(_data.size(), 1, base::Slice<T>(&_data));
        cuda::MatrixF64 m(t._data.size(), 1, base::Slice<T>(&t._data));
        auto rm = sm.pow(m);
        memcpy(res._data.data(), rm.getSlice().data(), rm.getSlice().size() * sizeof(T));
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            res._data[i] = ::pow(_data[i], t._data[i]);
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::transpose() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    auto shape = _shape.transpose();
    if (shape.tensorSize() <= 0) {
        return {};
    }
    Tensor<T> rm(shape);
    auto matrix_size = _shape.subTensorSize(_shape.dimCnt()-2);
    auto row_cnt = _shape.getDim(_shape.dimCnt() - 2);
    auto col_cnt = _shape.getDim(_shape.dimCnt() - 1);
    u32 index = 0;
    while (index * matrix_size < _data.size()) {
        if (std::is_same<T, f64>::value && ENABLE_CUDA) {
            cuda::MatrixF64 m(row_cnt, col_cnt, base::Slice<T>(&_data, index * matrix_size, matrix_size));
            auto tm = m.transpose();
            memcpy(rm._data.data() + index * matrix_size, tm.getSlice().data(), matrix_size * sizeof(T));
        } else {
            auto m = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data() + index * matrix_size, row_cnt, col_cnt);
            auto tm = m.transpose();
            memcpy(rm._data.data() + index * matrix_size, tm.data(), matrix_size * sizeof(T));
        }
        index += 1;
    }
    return std::move(rm);
}

template<typename T>
Tensor<T> Tensor<T>::mmul(const Tensor<T>& t) const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    auto shape = _shape.mmul(t._shape);
    if (shape.tensorSize() <= 0) {
        return {};
    }
    Tensor<T> rm(shape);

    auto row_cnt = _shape.getDim(_shape.dimCnt() - 2);
    auto col_cnt = _shape.getDim(_shape.dimCnt() - 1);
    auto matrix_size = _shape.subTensorSize(_shape.dimCnt() - 2);
    auto m_row_cnt = t._shape.getDim(t._shape.dimCnt() - 2);
    auto m_col_cnt = t._shape.getDim(t._shape.dimCnt() - 1);
    auto m_matrix_size = t._shape.subTensorSize(t._shape.dimCnt() - 2);
    auto r_matrix_size = rm._shape.subTensorSize(rm._shape.dimCnt() - 2);
    u32 index = 0;
    if (_shape.dimCnt() == 2) {
        while (index * m_matrix_size < t._data.size()) {
            if (std::is_same<T, f64>::value && ENABLE_CUDA) {
                cuda::MatrixF64 m1(row_cnt, col_cnt, base::Slice<T>(&_data, 0, matrix_size));
                cuda::MatrixF64 m2(m_row_cnt, m_col_cnt, base::Slice<T>(&t._data, index * m_matrix_size, m_matrix_size)); 
                auto m3 = m1.mmul(m2);
                memcpy(rm._data.data() + index * r_matrix_size, m3.getSlice().data(), r_matrix_size * sizeof(T));
            } else {
                auto m1 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data(), row_cnt, col_cnt);
                auto m2 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(t._data.data() + index * m_matrix_size, m_row_cnt, m_col_cnt);
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m3 = m1 * m2;
                memcpy(rm._data.data() + index * r_matrix_size, m3.data(), r_matrix_size * sizeof(T));
            }
            index += 1;
        }
    } else {
        while (index * matrix_size < _data.size()) {
            if (std::is_same<T, f64>::value && ENABLE_CUDA) {
                cuda::MatrixF64 m1(row_cnt, col_cnt, base::Slice<T>(&_data, index * matrix_size, matrix_size));
                cuda::MatrixF64 m2(m_row_cnt, m_col_cnt, base::Slice<T>(&t._data, 0, m_matrix_size));
                auto m3 = m1.mmul(m2);
                memcpy(rm._data.data() + index * r_matrix_size, m3.getSlice().data(), r_matrix_size * sizeof(T));
            } else {
                auto m1 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data() + index * matrix_size, row_cnt, col_cnt);
                auto m2 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(t._data.data(), m_row_cnt, m_col_cnt);
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m3 = m1 * m2;
                memcpy(rm._data.data() + index * r_matrix_size, m3.data(), r_matrix_size * sizeof(T));
            }
            index += 1;
        }
    }
    return std::move(rm);
}

template<typename T>
Tensor<T> Tensor<T>::inv() const {
    if (!this) {
        LOG(ERROR) << "tensor is null";
        return {};
    }
    if (_shape.dimCnt() < 2 || _shape.getDim(_shape.dimCnt() - 1) != _shape.getDim(_shape.dimCnt() - 2)) {
        LOG(ERROR) << "not square matrix";
        return {};
    }
    Tensor<T> rm(_shape);
    auto matrix_size = _shape.subTensorSize(_shape.dimCnt()-2);
    auto row_cnt = _shape.getDim(_shape.dimCnt() - 2);
    auto col_cnt = _shape.getDim(_shape.dimCnt() - 1);
    u32 index = 0;
    while (index * matrix_size < _data.size()) {
        if (std::is_same<T, f64>::value && ENABLE_CUDA) {
            cuda::MatrixF64 m(row_cnt, col_cnt, base::Slice<T>(&_data, index * matrix_size, matrix_size));
            auto tm = m.inv();
            memcpy(rm._data.data() + index * matrix_size, tm.getSlice().data(), matrix_size * sizeof(T));
        } else {
            auto m = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data() + index * matrix_size, row_cnt, col_cnt);
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tm = m.inverse();
            memcpy(rm._data.data() + index * matrix_size, tm.data(), matrix_size * sizeof(T));
        }
        index += 1;
    }
    return std::move(rm);
}

template<typename T>
const T& Tensor<T>::operator()(const std::vector<u32>& index) const {
    if (!checkIndex(index)) {
        return Tensor<T>::_INVALID_NUM;
    }
    u32 offset = calcOffset(index);
    return _data[offset];
}

template<typename T>
T& Tensor<T>::operator()(const std::vector<u32>& index) {
    if (!checkIndex(index)) {
        return Tensor<T>::_INVALID_NUM;
    }
    u32 offset = calcOffset(index);
    return _data[offset];
}

template<typename T>
bool Tensor<T>::assign(const std::vector<u32>& start_index, const std::vector<u32>& end_index, const T& e) {
    if (!checkIndex(start_index) || !checkIndex(end_index)) {
        LOG(ERROR) << "index invalid";
        return false;
    }
    u32 start_offset = calcOffset(start_index);
    u32 end_offset = calcOffset(end_index);
    if (start_index > end_index) {
        LOG(ERROR) << "start index gt end index";
        return false;
    }
    for (int i = start_offset; i <= end_offset; i++) {
        _data[i] = e;
    }
    return true;
}

template<typename T>
std::string Tensor<T>::toString(bool compact) const {
    if (!this) {
        return "null";
    }
    std::function<std::string(u32, u32&)> recur_gen = [&recur_gen, &compact, this](u32 dim_index, u32& offset) -> std::string {
        std::stringstream stream;
        stream << std::string(dim_index*4, ' ') << "[";
        if (dim_index == _shape.dimCnt() - 1) {
            for (int i = 0; i < _shape.getDim(dim_index); i++) {
                stream << _data[offset];
                if (i != _shape.getDim(dim_index) - 1) {
                    stream << ", ";
                }
                offset += 1;
            }
            stream << "]";
            if (!compact) {
                stream << "\n";
            }
        } else {
            if (!compact) {
                stream << "\n";
            }
            for (int i = 0; i < _shape.getDim(dim_index); i++) {
                stream << recur_gen(dim_index + 1, offset);
            }
            stream << std::string(dim_index*4, ' ') << "]";
            if (!compact) {
                stream << "\n";
            }
        }
        return stream.str();
    };
    u32 offset = 0;
    return recur_gen(0, offset);
}

template<typename T>
const std::vector<T>& Tensor<T>::getData() const {
    return _data;
}

template<typename T>
bool Tensor<T>::checkIndex(const std::vector<u32>& index) const {
    if (index.size() != _shape.dimCnt()) {
        LOG(ERROR) << "index size != dim cnt";
        return false;
    }
    for (int i = 0; i < _shape.dimCnt(); i++) {
        if (index[i] >= _shape.getDim(i)) {
            LOG(ERROR) << "index out of dim range, index: " << index[i] 
                << ", dim: " << _shape.getDim(i);
            return false;
        }        
    }
    return true;
}

template<typename T>
u32 Tensor<T>::calcOffset(const std::vector<u32>& index) const {
    u32 offset = 0;
    for (int i = 0; i < _shape.dimCnt(); i++) {
        if (i + 1 == _shape.dimCnt()) {
            offset += index[i];
        } else {
            offset += index[i] * _shape.subTensorSize(i + 1);
        }
    }
    return offset;
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<u32>& dims) const {
    auto shape = _shape.reshape(dims);
    if (shape.tensorSize() != _shape.tensorSize()) {
        LOG(ERROR) << "reshape fail";
        return Tensor();
    }
    return Tensor(shape, _data);
}

template<typename T>
bool Tensor<T>::reshapeInplace(const std::vector<u32>& dims) {
    auto shape = _shape.reshape(dims);
    if (shape.tensorSize() != _shape.tensorSize()) {
        LOG(ERROR) << "reshape fail";
        return false;
    }
    _shape = std::move(shape);
    return true;
}

}

#endif