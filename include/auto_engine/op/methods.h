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
T transpose(const T&);

template<typename T>
T mmul(const T&, const T&);

template<typename T>
T inv(const T&);

}
#endif