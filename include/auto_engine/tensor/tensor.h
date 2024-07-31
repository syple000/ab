#ifndef BASE_TENSOR_H 
#define BASE_TENSOR_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/cuda/tensor.h"
#include "auto_engine/shape/shape.h"
#include "glog/logging.h"
#include "Eigen/Dense"
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fmt/core.h>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace base {

template<typename T>
struct Tensor {
public:
    Tensor() {}
    Tensor(const Shape& shape): _shape(shape), _data(std::vector<T>(shape.tensorSize())) {
        if (_shape.tensorSize() == 0) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error("tensor size 0");
            }
            LOG(ERROR) << "tensor size 0";
            _shape = Shape();
            _data = std::vector<T>();
        }
    }
    Tensor(const Shape& shape, const std::function<T()>& f): _shape(shape), _data(std::vector<T>(shape.tensorSize())) {
        if (_shape.tensorSize() == 0) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error("tensor size 0");
            }
            LOG(ERROR) << "tensor size 0";
            _shape = Shape();
            _data = std::vector<T>();
        }
        for (auto& d : _data) {
            d = f();
        }
    }
    Tensor(const Shape& shape, const T& e): _shape(shape), _data(std::vector<T>(shape.tensorSize(), e)) {
        if (_shape.tensorSize() == 0) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error("tensor size 0");
            }
            LOG(ERROR) << "tensor size 0";
            _shape = Shape();
            _data = std::vector<T>();
        }
    }
    Tensor(const Shape& shape, const std::vector<T>& data): _shape(shape), _data(std::vector<T>(data)) {
        if (_shape.tensorSize() != _data.size()) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error(fmt::format("tensor size invalid, shape size: {}, data size: {}", shape.tensorSize(), data.size()));
            }
            LOG(ERROR) << "tensor size invalid";
            _shape = Shape();
            _data = std::vector<T>();
        }
        if (_shape.tensorSize() == 0) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error("tensor size 0");
            }
            LOG(ERROR) << "tensor size 0";
            _shape = Shape();
            _data = std::vector<T>();
        }
    }

    Tensor(const Tensor<T>& t) = default;
    Tensor(Tensor<T>&&) = default;
    Tensor<T>& operator=(const Tensor<T>&) = default;
    Tensor<T>& operator=(Tensor<T>&&) = default;

    bool operator==(const Tensor<T>& t) const {
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

    Tensor<T> operator+(const Tensor<T>&) const;
    Tensor<T> operator+(T) const;
    Tensor<T> operator-(const Tensor<T>&) const;
    Tensor<T> operator-(T) const;
    Tensor<T> operator*(const Tensor<T>&) const;
    Tensor<T> operator*(T) const;
    Tensor<T> operator/(const Tensor<T>&) const;
    Tensor<T> operator/(T) const;

    Tensor<T> neg() const;
    Tensor<T> log() const;
    Tensor<T> sin() const;
    Tensor<T> cos() const;
    Tensor<T> sign() const;
    Tensor<T> abs() const;
    Tensor<T> pow(const Tensor<T>&) const;
    Tensor<T> pow(T) const;

    Tensor<T> transpose(int d1, int d2) const;
    Tensor<T> mmul(const Tensor<T>&) const;
    Tensor<T> inv() const;

    Tensor<T> sum() const;
    Tensor<T> expand(const base::Shape&) const;
    Tensor<T> sum(int d) const;
    Tensor<T> expand(int d, u32 expd) const;

    Tensor<T> oneHot(u32 classes) const;

    Tensor<T> reshape(const std::vector<u32>& dims) const {
        auto shape = _shape.reshape(dims);
        if (shape.tensorSize() != _shape.tensorSize()) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error(fmt::format("tensor reshape err, orig shape: {}, target shape: {}", _shape.toString(), shape.toString()));
            }
            LOG(ERROR) << "tensor reshape err";
            return Tensor();
        }
        return Tensor(shape, _data);
    }
    Tensor<T> reshape(const base::Shape& shape) const {
        if (shape.tensorSize() != _shape.tensorSize()) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error(fmt::format("tensor reshape err, orig shape: {}, target shape: {}", _shape.toString(), shape.toString()));
            }
            LOG(ERROR) << "tensor reshape err";
            return Tensor();
        }
        return Tensor(shape, _data);
    }

    std::string toString(bool compact = false) const {
        std::function<std::string(u32, u32&)> recur_gen = [&recur_gen, &compact, this](u32 dim_index, u32& offset) -> std::string {
            std::stringstream stream;
            if (!compact) {
                stream << std::string(dim_index*4, ' ');
            }
            stream << "[";
            if (dim_index == _shape.dimCnt() - 1) {
                for (int i = 0; i < _shape.getDim(dim_index); i++) {
                    stream << std::fixed << std::setprecision(9) << _data[offset];
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
                if (!compact) {
                    stream << std::string(dim_index*4, ' ');
                }
                stream << "]";
                if (!compact) {
                    stream << "\n";
                }
            }
            return stream.str();
        };
        u32 offset = 0;
        return recur_gen(0, offset);
    }

    const Shape& shape() const {return _shape;}
    const std::vector<T> data() const {return _data;}
private:
    Shape _shape;
    std::vector<T> _data;
};


template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& t) const {
    if (!(_shape == t._shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor + fail, shape diff, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        LOG(ERROR) << "tensor + fail, shape diff";
        return Tensor();
    }
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_add(ret._data.data(), t._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            ret._data[i] += t._data[i];
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T t) const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_add(ret._data.data(), t, ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] += t;
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& t) const {
    if (!(_shape == t._shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor - fail, shape diff, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        LOG(ERROR) << "tensor - fail, shape diff";
        return Tensor();
    }
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_sub(ret._data.data(), t._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            ret._data[i] -= t._data[i];
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(T t) const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_sub(ret._data.data(), t, ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] -= t;
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& t) const {
    if (!(_shape == t._shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor * fail, shape diff, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        LOG(ERROR) << "tensor * fail, shape diff";
        return Tensor();
    }
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_mul(ret._data.data(), t._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            ret._data[i] *= t._data[i];
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T t) const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_mul(ret._data.data(), t, ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] *= t;
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& t) const {
    if (!(_shape == t._shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor / fail, shape diff, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        LOG(ERROR) << "tensor / fail, shape diff";
        return Tensor();
    }
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_div(ret._data.data(), t._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            ret._data[i] /= t._data[i];
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T t) const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_div(ret._data.data(), t, ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] /= t;
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T>& t) const {
    if (!(_shape == t._shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor pow fail, shape diff, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        LOG(ERROR) << "tensor pow fail, shape diff";
        return Tensor();
    }
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_pow(ret._data.data(), t._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < t._data.size(); i++) {
            ret._data[i] = ::pow(ret._data[i], t._data[i]);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::pow(T t) const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_pow(ret._data.data(), t, ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = ::pow(ret._data[i], t);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::neg() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_neg(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = -ret._data[i];
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::log() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_log(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = ::log(ret._data[i]);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::sign() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_sign(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            if (ret._data[i] >= 0) {
                ret._data[i] = 1;
            } else {
                ret._data[i] = -1;
            }
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::abs() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_abs(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = ::abs(ret._data[i]);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::sin() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_sin(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = ::sin(ret._data[i]);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::cos() const {
    auto ret = *this;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::apply_cos(ret._data.data(), ret._data.size());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            ret._data[i] = ::cos(ret._data[i]);
        }
    }
    return ret;
}

template<typename T>
Tensor<T> Tensor<T>::transpose(int d1, int d2) const {
    if (d1 < 0) {d1 = _shape.dimCnt() + d1;}
    if (d2 < 0) {d2 = _shape.dimCnt() + d2;}

    auto shape = _shape.transpose(d1, d2);
    if (shape.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor transpose fail: {}", _shape.toString()));
        }
        return Tensor();
    }

    auto rt = Tensor(shape, _data); // 仅对shape进行转置，数据还未转置

    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::transpose(rt._data.data(), _shape.getDims(), d1, d2);
    } else {
        std::vector<u32> index(_shape.dimCnt());
        for (u32 i = 0; i < _shape.tensorSize(); i++) { 
            u32 ri = i;
            for (int j = 0; j < _shape.dimCnt(); j++) {
                index[j] = ri / _shape.getStrides()[j];
                ri = ri % _shape.getStrides()[j];
            }

            std::swap(index[d1], index[d2]);
            u32 oi = 0;
            for (int j = 0; j < shape.dimCnt(); j++) {
                oi += index[j] * shape.getStrides()[j];
            }
            rt._data[oi] = _data[i];
        }
    }
    return std::move(rt);
}

template<typename T>
Tensor<T> Tensor<T>::mmul(const Tensor<T>& t) const {
    auto shape = _shape.mmul(t._shape);
    if (shape.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor mmul fail, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        return Tensor();
    }
    auto rt = Tensor(shape);

    auto row_cnt = _shape.getDim(_shape.dimCnt() - 2);
    auto col_cnt = _shape.getDim(_shape.dimCnt() - 1);
    auto matrix_size = _shape.subTensorSize(_shape.dimCnt() - 2);
    auto m_row_cnt = t._shape.getDim(t._shape.dimCnt() - 2);
    auto m_col_cnt = t._shape.getDim(t._shape.dimCnt() - 1);
    auto m_matrix_size = t._shape.subTensorSize(t._shape.dimCnt() - 2);
    auto r_matrix_size = rt._shape.subTensorSize(rt._shape.dimCnt() - 2);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::mmul(row_cnt, m_col_cnt, col_cnt, _data.data(), t._data.data(), rt._data.data(), rt._shape.tensorSize()/r_matrix_size);
    } else {
        int index = 0;
        while (index * matrix_size < _data.size()) {
            auto m1 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data() + index * matrix_size, row_cnt, col_cnt);
            auto m2 = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(t._data.data() + index * m_matrix_size, m_row_cnt, m_col_cnt);
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m3 = m1 * m2;
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rt._data.data() + index * r_matrix_size, row_cnt, m_col_cnt) = m3;
            index += 1;
        }
    }
    return std::move(rt);
}

template<typename T>
Tensor<T> Tensor<T>::inv() const {
    if (_shape.dimCnt() < 2 || _shape.getDim(_shape.dimCnt() - 2) != _shape.getDim(_shape.dimCnt() - 1)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor inv fail: {}", _shape.toString()));
        }
        LOG(ERROR) << "not a square matrix";
        return Tensor();
    }
    auto rt = *this;
    auto row_col_cnt = _shape.getDim(_shape.dimCnt() - 1);
    auto matrix_size = _shape.subTensorSize(_shape.dimCnt() - 2);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        if (!cuda::inv(row_col_cnt, rt._data.data(), _shape.tensorSize()/matrix_size)) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error(fmt::format("tensor inv fail"));
            }
            rt = Tensor();
        }
    } else {
        int index = 0;
        while (index * matrix_size < _data.size()) {
            auto m = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(_data.data() + index * matrix_size, row_col_cnt, row_col_cnt);
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tm = m.inverse();
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(rt._data.data() + index * matrix_size, row_col_cnt, row_col_cnt) = tm;
            index += 1;
        }
    }
    return std::move(rt);
}

template<typename T>
Tensor<T> Tensor<T>::sum() const {
    auto shape = _shape.sum();
    if (shape.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor sum fail, shape: {}", _shape.toString()));
        }
        LOG(ERROR) << fmt::format("tensor sum fail, shape: {}", _shape.toString());
        return Tensor();
    }
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::sum(_data.data(), _data.size(), &res._data[0]);
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[0] += _data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::expand(const base::Shape& shape) const {
    auto sp = _shape.expand(shape);
    if (sp.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("expand tensor fail: {}, {}", _shape.toString(), shape.toString()));
        }
        LOG(ERROR) << fmt::format("expand tensor fail: {}, {}", _shape.toString(), shape.toString());
        return Tensor();
    }
    Tensor<T> res(sp);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::expand(_data.data(), res._data.data(), res._data.size()); 
    } else {
        res = Tensor<T>(sp, _data[0]);
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::sum(int d) const {
    if (d < 0) {d = d + _shape.dimCnt();}

    auto shape = _shape.sum(d);
    if (shape.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor sum fail, shape: {}, d: {}", _shape.toString(), d));
        }
        LOG(ERROR) << fmt::format("tensor sum fail, shape: {}, d: {}", _shape.toString(), d);
        return Tensor();
    }
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::sum(_data.data(), res._data.data(), _shape, d);
    } else {
        std::vector<u32> index(shape.dimCnt());
        for (int i = 0; i < _data.size(); i++) {
            u32 ri = i;
            for (int j = 0; j < _shape.dimCnt(); j++) {
                if (j < d) {
                    index[j] = ri / _shape.getStrides()[j];
                } else if (j > d) {
                    index[j - 1] = ri / _shape.getStrides()[j];
                }
                ri = ri % _shape.getStrides()[j];
            }
            u32 oi = 0;
            for (int j = 0; j < shape.dimCnt(); j++) {
                oi = oi + shape.getStrides()[j] * index[j];
            }
            res._data[oi] += _data[i]; 
        }
    }
    return std::move(res);
}

template<typename T>    
Tensor<T> Tensor<T>::expand(int d, u32 expd) const {
    if (d < 0) {d = d + _shape.dimCnt() + 1;}

    auto shape = _shape.expand(d, expd);
    if (shape.tensorSize() <= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor expand fail, shape: {}, d: {}, expd: {}", _shape.toString(), d, expd));
        }
        LOG(ERROR) << fmt::format("tensor expand fail, shape: {}, d: {}, expd: {}", _shape.toString(), d, expd);
        return Tensor();
    }
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::expand(_data.data(), res._data.data(), _shape, d, expd);
    } else {
        std::vector<u32> index(_shape.dimCnt());
        for (int i = 0; i < res._data.size(); i++) {
            u32 ri = i;
            for (int j = 0; j < res._shape.dimCnt(); j++) {
                if (j < d) {
                    index[j] = ri / res._shape.getStrides()[j];
                } else if (j > d) {
                    index[j - 1] = ri / res._shape.getStrides()[j];
                }
                ri = ri % res._shape.getStrides()[j];
            }
            u32 ii = 0;
            for (int j = 0; j < _shape.dimCnt(); j++) {
                ii = ii + _shape.getStrides()[j] * index[j];
            }
            res._data[i] += _data[ii]; 
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::oneHot(u32 classes) const {
    if (_shape.dimCnt() != 1) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("one hot tensor dim cnt != 1");
        }
        LOG(ERROR) << "one hot tensor dim cnt != 1";
        return Tensor();
    }
    auto res = Tensor<T>(base::Shape({_shape.getDim(0), classes}));
    int err_index = -1;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        err_index = cuda::ont_hot(_data.data(), _shape.tensorSize(), res._data.data(), classes);
    } else {
        for (int i = 0; i < _data.size(); i++) {
            int n = round(_data[i]);
            if (n < 0 || n >= classes) {err_index = i; break;}
            res._data[i * classes + n] = 1;
        }
    }
    if (err_index >= 0) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("one hot fail at {}", err_index));
        }
        LOG(ERROR) << fmt::format("one hot fail at {}", err_index);
        return Tensor();
    }
    return std::move(res);
}

}

#endif