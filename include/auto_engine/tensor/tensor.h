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
    Tensor(const Shape& shape): _shape(shape), _data(std::vector<T>(shape.tensorSize())) {}
    Tensor(const Shape& shape, const std::function<T()>& f): _shape(shape), _data(std::vector<T>(shape.tensorSize())) {
        for (auto& d : _data) {
            d = f();
        }
    }
    Tensor(const Shape& shape, const T& e): _shape(shape), _data(std::vector<T>(shape.tensorSize(), e)) {}
    Tensor(const Shape& shape, const std::vector<T>& data): _shape(shape), _data(std::vector<T>(data)) {
        if (_shape.tensorSize() != _data.size()) {
            if (ENABLE_TENSOR_EXCEPTION) {
                throw std::runtime_error(fmt::format("tensor size invalid, shape size: {}, data size: {}", shape.tensorSize(), data.size()));
            }
            LOG(ERROR) << "tensor size invalid";
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

    Tensor<T> permute(const std::vector<u32>& idxs) const;
    Tensor<T> transpose(int d1, int d2) const;
    Tensor<T> mmul(const Tensor<T>&) const;
    Tensor<T> inv() const;

    Tensor<T> sum() const;
    Tensor<T> expand(const base::Shape&) const;
    Tensor<T> sum(int d) const;
    Tensor<T> expand(const base::Shape&, int d) const;
    
    static Tensor<T> cat(const std::vector<std::reference_wrapper<Tensor<T>>>& ts, int d);
    static std::vector<Tensor<T>> split(const Tensor<T>&, const std::vector<u32>& sl, int d);
    Tensor<T> cat(const Shape& dst_shape, int d, u32 d_offset);
    Tensor<T> split(const Shape& dst_shape, int d, u32 d_offset);

    Tensor<T> oneHot(u32 classes) const;

    Tensor<T> reshape(const std::vector<u32>& dims) const {
        Shape shape;
        if (!_shape.reshape(dims, shape)) {
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
    if (_shape.tensorSize() == 0) {return *this;}
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
Tensor<T> Tensor<T>::permute(const std::vector<u32>& pl) const {
    Shape shape;
    if (!_shape.permute(pl, shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("tensor permute fail");
        }
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor<T>(shape);}
    // pl重映射，由原下标获取新下标
    std::vector<u32> npl(pl.size());
    for (u32 i = 0; i < pl.size(); i++) {
        npl[pl[i]] = i;
    }
    auto rt = Tensor(shape, _data); // 仅对shape进行转置，数据还未转置
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::permute(_data.data(), _shape, rt._data.data(), rt._shape, npl);
    } else {
        for (u32 i = 0; i < _shape.tensorSize(); i++) { 
            u32 ri = i, oi = 0;
            for (int j = 0; j < _shape.dimCnt(); j++) {
                oi += (ri / _shape.getStrides()[j]) * rt._shape.getStrides()[npl[j]];
                ri = ri % _shape.getStrides()[j];
            }
            rt._data[oi] = _data[i];
        }
    }
    return std::move(rt);
}

template<typename T>
Tensor<T> Tensor<T>::transpose(int d1, int d2) const {
    if (d1 < 0) {d1 = _shape.dimCnt() + d1;}
    if (d2 < 0) {d2 = _shape.dimCnt() + d2;}

    Shape shape;
    if (!_shape.transpose(d1, d2, shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor transpose fail: {}", _shape.toString()));
        }
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor(shape);}

    std::vector<u32> pl(shape.dimCnt());
    for (u32 i = 0; i < pl.size(); i++) {pl[i] = i;}
    std::swap(pl[d1], pl[d2]);

    auto rt = Tensor(shape, _data); // 仅对shape进行转置，数据还未转置

    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::permute(_data.data(), _shape, rt._data.data(), rt._shape, pl);
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
    Shape shape;
    if (!_shape.mmul(t._shape, shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor mmul fail, shape1: {}, shape2: {}", _shape.toString(), t._shape.toString()));
        }
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor(shape);}

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
    if (_shape.tensorSize() == 0) {return Tensor(_shape);}
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
    Shape shape;
    if (!_shape.sum(shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor sum fail, shape: {}", _shape.toString()));
        }
        LOG(ERROR) << fmt::format("tensor sum fail, shape: {}", _shape.toString());
        return Tensor();
    }
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::sum(_data.data(), _data.size(), res._data.data());
    } else {
        for (int i = 0; i < _data.size(); i++) {
            res._data[0] += _data[i];
        }
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::expand(const base::Shape& shape) const {
    if (!_shape.expand(shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("expand tensor fail: {}, {}", _shape.toString(), shape.toString()));
        }
        LOG(ERROR) << fmt::format("expand tensor fail: {}, {}", _shape.toString(), shape.toString());
        return Tensor();
    }
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::expand(_data.data(), res._data.data(), res._data.size()); 
    } else {
        res = Tensor<T>(shape, _data[0]);
    }
    return std::move(res);
}

template<typename T>
Tensor<T> Tensor<T>::sum(int d) const {
    if (d < 0) {d = d + _shape.dimCnt();}

    Shape shape;
    if (!_shape.sum(d, shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor sum fail, shape: {}, d: {}", _shape.toString(), d));
        }
        LOG(ERROR) << fmt::format("tensor sum fail, shape: {}, d: {}", _shape.toString(), d);
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor(shape);}
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::sum(_data.data(), _shape, res._data.data(), d);
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
Tensor<T> Tensor<T>::expand(const Shape& shape, int d) const {
    if (d < 0) {d = d + _shape.dimCnt() + 1;}

    if (!_shape.expand(shape, d)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("tensor expand fail, shape: {}, org shape: {}, d: {}", shape.toString(), _shape.toString(), d));
        }
        LOG(ERROR) << fmt::format("tensor expand fail, shape: {}, org shape: {}", shape.toString(), _shape.toString());
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor(shape);}
    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::expand(_data.data(), res._data.data(), shape, d);
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
Tensor<T> Tensor<T>::cat(const Shape& dst_shape, int d, u32 d_offset) {
    if (d < 0) {d += dst_shape.dimCnt();}

    if (!_shape.cat(dst_shape, d, d_offset)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("cat src to dst fail");
        }
        LOG(ERROR) << "cat src to dst fail";
        return Tensor();
    }
    auto dst = Tensor<T>(dst_shape);
    if (shape().tensorSize() == 0) {return std::move(dst);}
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::cat(_data.data(), shape(), dst._data.data(), dst.shape(), d, d_offset);
    } else {
        std::vector<u32> indexs(dst.shape().dimCnt());
        for (u32 i = 0; i < _data.size(); i++) {
            u32 ri = i;
            for (u32 j = 0; j < dst.shape().dimCnt(); j++) {
                auto dim_index = ri / shape().getStrides()[j];
                if (j == d) {
                    dim_index += d_offset;
                }
                ri = ri % shape().getStrides()[j];
                indexs[j] = dim_index;
            }
            auto didx = 0;
            for (u32 j = 0; j < dst.shape().dimCnt(); j++) {
                didx += indexs[j] * dst.shape().getStrides()[j];
            }
            dst._data[didx] = _data[i];
        }
    }
    return std::move(dst);
}

template<typename T>    
Tensor<T> Tensor<T>::split(const Shape& dst_shape, int d, u32 d_offset) {
    if (d < 0) {d += dst_shape.dimCnt();}

    if (!_shape.split(dst_shape, d, d_offset)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("split to dst fail");
        }
        LOG(ERROR) << "split to dst fail";
        return Tensor();
    }
    auto dst = Tensor(dst_shape);
    if (dst.shape().tensorSize() == 0) {return std::move(dst);}
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::split(_data.data(), shape(), dst._data.data(), dst.shape(), d, d_offset);
    } else {
        std::vector<u32> indexs(dst.shape().dimCnt());
        for (u32 i = 0; i < dst._data.size(); i++) {
            u32 ri = i;
            for (u32 j = 0; j < dst.shape().dimCnt(); j++) {
                auto dim_index = ri / dst.shape().getStrides()[j];
                if (j == d) {
                    dim_index += d_offset;
                }
                ri = ri % dst.shape().getStrides()[j];
                indexs[j] = dim_index;
            }
            u32 sidx = 0;
            for (u32 j = 0; j < shape().dimCnt(); j++) {
                sidx += indexs[j] * shape().getStrides()[j];
            }
            dst._data[i] = _data[sidx];
        }
    }
    return std::move(dst);
}

template<typename T>
Tensor<T> Tensor<T>::cat(const std::vector<std::reference_wrapper<Tensor<T>>>& ts, int d) {
    std::vector<std::reference_wrapper<Shape>> ss; ss.reserve(ts.size());
    std::vector<const f64*> srcs; srcs.reserve(ts.size());
    int dim_cnt = 0;
    for (u32 i = 0; i < ts.size(); i++) {
        auto s = ts[i].get();
        ss.emplace_back(s._shape);
        srcs.emplace_back(s._data.data());
        dim_cnt = s.shape().dimCnt();
    }
    if (d < 0) {d += dim_cnt;}
    Shape shape;
    if (!Shape::cat(ss, d, shape)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("cat tensor fail");
        }
        LOG(ERROR) << "cat tensor fail";
        return Tensor();
    }
    if (shape.tensorSize() == 0) {return Tensor(shape);}

    Tensor<T> res(shape);
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::cat(srcs, ss, res._data.data(), res._shape, d);  
    } else {
        std::vector<u32> indexs(res._shape.dimCnt());
        u32 src_index = 0;
        for (u32 i = 0; i < res._data.size(); i++) {
            auto ri = i;
            for (u32 j = 0; j < res._shape.dimCnt(); j++) {
                auto dim_index = ri / res._shape.getStrides()[j];
                if (j == d) {
                    for (u32 k = 0; k < ts.size(); k++) {
                        if (dim_index >= ts[k].get()._shape.getDim(d)) {
                            dim_index -= ts[k].get()._shape.getDim(d);
                        } else {
                            src_index = k;
                            break;
                        }
                    }
                }
                indexs[j] = dim_index;
                ri = ri % res._shape.getStrides()[j];
            }
            u32 sidx = 0;
            for (u32 j = 0; j < res._shape.dimCnt(); j++) {
                sidx += indexs[j] * ts[src_index].get()._shape.getStrides()[j];
            }
            res._data[i] = ts[src_index].get()._data[sidx];
        }
    }
    return std::move(res);
}

template<typename T>
std::vector<Tensor<T>> Tensor<T>::split(const Tensor<T>& t, const std::vector<u32>& sl, int d) {
    if (d < 0) {d += t.shape().dimCnt();}

    std::vector<Shape> ss;
    if (!Shape::split(t._shape, sl, d, ss)) {
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error("split tensor fail");
        }
        LOG(ERROR) << "split tensor fail";
        return {};
    }
    std::vector<Tensor<T>> dts; dts.reserve(ss.size());
    std::vector<f64*> dsts; dsts.reserve(ss.size());
    std::vector<std::reference_wrapper<Shape>> dtss; dtss.reserve(ss.size());
    for (u32 i = 0; i < ss.size(); i++) {
        dts.emplace_back(Tensor(ss[i]));
        dsts.emplace_back(dts[i]._data.data());
        dtss.emplace_back(ss[i]);
    }
    if (t.shape().tensorSize() == 0) {return std::move(dts);}

    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        cuda::split(t._data.data(), t.shape(), dsts, dtss, d);  
    } else {
        std::vector<u32> indexs(t._shape.dimCnt());
        u32 dst_index = 0;
        for (u32 i = 0; i < t._data.size(); i++) {
            auto ri = i;
            for (u32 j = 0; j < t._shape.dimCnt(); j++) {
                auto dim_index = ri / t._shape.getStrides()[j];
                if (j == d) {
                    for (u32 k = 0; k < ss.size(); k++) {
                        if (dim_index >= ss[k].getDim(d)) {
                            dim_index -= ss[k].getDim(d);
                        } else {
                            dst_index = k;
                            break;
                        }
                    }
                }
                indexs[j] = dim_index;
                ri = ri % t._shape.getStrides()[j];
            }
            u32 didx = 0;
            for (u32 j = 0; j < t._shape.dimCnt(); j++) {
                didx += indexs[j] * ss[dst_index].getStrides()[j];
            }
            dts[dst_index]._data[didx] = t._data[i];
        }
    }
    return std::move(dts);
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
    if (res.shape().tensorSize() == 0) {return res;}
    u32 err_occur = 0, err_index = 0;
    if (std::is_same<T, f64>::value && ENABLE_CUDA) {
        u32 err_occur = 0, err_index = 0;
        cuda::one_hot(_data.data(), _data.size(), res._data.data(), classes, &err_occur, &err_index);
    } else {
        for (int i = 0; i < _data.size(); i++) {
            int n = round(_data[i]);
            if (n < 0 || n >= classes) {err_occur = 1; err_index = i; break;}
            res._data[i * classes + n] = 1;
        }
    }
    if (err_occur) {
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