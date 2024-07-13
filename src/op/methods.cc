#include "auto_engine/op/methods.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>

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

}