#ifndef UTILS_TUPLE_UTILS_H
#define UTILS_TUPLE_UTILS_H

#include <tuple>

template<int N, typename T, typename... TS>
struct GenTup {
    using type = typename GenTup<N-1, T, T, TS...>::type;
};

template<typename T, typename... TS>
struct GenTup<0, T, TS...> {
    using type = std::tuple<TS...>;
};

template<int N, typename T>
using GenTupT = typename GenTup<N, T>::type;

#endif