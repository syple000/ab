#include "auto_engine/op/methods.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/shape/shape.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <fmt/core.h>
#include <stdexcept>

namespace op {

template<>
f64 zero<f64>(const f64&) {
    return 0;
}
template<>
base::Tensor<f64> zero<base::Tensor<f64>>(const base::Tensor<f64>& t) {
    return base::Tensor<f64>(t.shape(), 0);
}

template<>
f64 one<f64>(const f64&) {
    return 1;
}
template<>
base::Tensor<f64> one<base::Tensor<f64>>(const base::Tensor<f64>& t) {
    return base::Tensor<f64>(t.shape(), 1);
}

template<>
f64 sin<f64>(const f64& n) {
    return ::sin(n);
}
template<>
base::Tensor<f64> sin<base::Tensor<f64>>(const base::Tensor<f64>& t) {
    return t.sin();
}

template<>
f64 cos<f64>(const f64& n) {
    return ::cos(n);
}
template<>
base::Tensor<f64> cos<base::Tensor<f64>>(const base::Tensor<f64>& t) {
    return t.cos();
}

template<>
f64 pow<f64>(const f64& n1, const f64& n2) {
    return ::pow(n1, n2);
}
template<>
base::Tensor<f64> pow<base::Tensor<f64>>(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    return t1.pow(t2);
}

template<>
f64 log<f64>(const f64& n) {
    return ::log(n);    
}
template<>
base::Tensor<f64> log<base::Tensor<f64>>(const base::Tensor<f64>& t) {
    return t.log();
}

template<>
base::Tensor<f64> transpose(const base::Tensor<f64>& t, int d1, int d2) {
    return t.transpose(d1, d2);
}

template<>
base::Tensor<f64> permute(const base::Tensor<f64>& t, const std::vector<u32>& pl) {
    return t.permute(pl);
}

template<>
base::Tensor<f64> mmul(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    return t1.mmul(t2);
}

template<>
base::Tensor<f64> inv(const base::Tensor<f64>& t) {
    return t.inv();
}

template<>
base::Tensor<f64> reshape(const base::Tensor<f64>& t, const base::Shape& shape) {
    return t.reshape(shape);
}

template<>
base::Tensor<f64> sum(const base::Tensor<f64>& t) {
    return t.sum();
}

template<>
base::Tensor<f64> expand(const base::Tensor<f64>& t, const base::Shape& shape) {
    return t.expand(shape);
}

template<>
base::Tensor<f64> sum(const base::Tensor<f64>& t, int d) {
    return t.sum(d);
}

template<>
base::Tensor<f64> expand(const base::Tensor<f64>& t, const base::Shape& shape, int d) {
    return t.expand(shape, d);
}

template<>
base::Shape shape(const base::Tensor<f64>& t) {
    return t.shape();
}


template<>
base::Tensor<f64> add_n(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    if (t2.shape().tensorSize() != 1) {
        LOG(ERROR) << fmt::format("[{}] t2 tensor size != 1", __FUNCTION__);
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("[{}] t2 tensor size != 1", __FUNCTION__));
        }
        return base::Tensor<f64>();
    }
    return t1 + t2.data()[0];
}

template<>
base::Tensor<f64> sub_n(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    if (t2.shape().tensorSize() != 1) {
        LOG(INFO) << fmt::format("[{}] t2 tensor size != 1", __FUNCTION__);
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("[{}] t2 tensor size != 1", __FUNCTION__));
        }
        return base::Tensor<f64>();
    }
    return t1 - t2.data()[0];
}

template<>
base::Tensor<f64> mul_n(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    if (t2.shape().tensorSize() != 1) {
        LOG(INFO) << fmt::format("[{}] t2 tensor size != 1", __FUNCTION__);
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("[{}] t2 tensor size != 1", __FUNCTION__));
        }
        return base::Tensor<f64>();
    }
    return t1 * t2.data()[0];
}

template<>
base::Tensor<f64> div_n(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    if (t2.shape().tensorSize() != 1) {
        LOG(INFO) << fmt::format("[{}] t2 tensor size != 1", __FUNCTION__);
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("[{}] t2 tensor size != 1", __FUNCTION__));
        }
        return base::Tensor<f64>();
    }
    return t1 / t2.data()[0];
}

template<>
base::Tensor<f64> pow_n(const base::Tensor<f64>& t1, const base::Tensor<f64>& t2) {
    if (t2.shape().tensorSize() != 1) {
        LOG(INFO) << fmt::format("[{}] t2 tensor size != 1", __FUNCTION__);
        if (ENABLE_TENSOR_EXCEPTION) {
            throw std::runtime_error(fmt::format("[{}] t2 tensor size != 1", __FUNCTION__));
        }
        return base::Tensor<f64>();
    }
    return t1.pow(t2.data()[0]);
}

template<>
base::Tensor<f64> cat(const std::vector<std::reference_wrapper<const base::Tensor<f64>>>& ts, int d) {
    return base::Tensor<f64>::cat(ts, d);
}

template<>
std::vector<base::Tensor<f64>> split(const base::Tensor<f64>& t, const std::vector<u32>& sl, int d) {
    return base::Tensor<f64>::split(t, sl, d);
}


}