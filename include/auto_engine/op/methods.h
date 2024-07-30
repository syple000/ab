#ifndef OP_METHODS_H
#define OP_METHODS_H

namespace op {

template<typename T>
T zero(const T&); // 零元

template<typename T>
T one(const T&); // 1元

template<typename T>
T sin(const T&);

template<typename T>
T cos(const T&);

template<typename T>
T pow(const T&, const T&);

template<typename T>
T log(const T&);

template<typename T>
T transpose(const T&, int, int);

template<typename T>
T mmul(const T&, const T&);

template<typename T>
T inv(const T&);

template<typename T>
T sum(const T&);

template<typename T, typename SHAPE>
T expand(const T&, const SHAPE&);

template<typename T>
T sum_along_col(const T&);

template<typename T, typename SHAPE>
T expand_along_col(const T&, const SHAPE&);

template<typename T>
T sum_along_row(const T&);

template<typename T, typename SHAPE>
T expand_along_row(const T&, const SHAPE&);

template<typename T, typename SHAPE>
T reshape(const T&, const SHAPE&);

template<typename T, typename SHAPE>
SHAPE shape(const T&);

template<typename T>
T add_n(const T&, const T&);

template<typename T>
T sub_n(const T&, const T&);

template<typename T>
T mul_n(const T&, const T&);

template<typename T>
T div_n(const T&, const T&);

template<typename T>
T pow_n(const T&, const T&);

}
#endif