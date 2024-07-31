#ifndef OP_CAT_SPLIT_H
#define OP_CAT_SPLIT_H

#include "auto_engine/op/op.h"

namespace op {

// 特殊的操作，操作是多（>=2）到1，或1到多
// 并且参数间隔离，uop/bop的模式不再适用，直接继承op

template<typename T, typename SHAPE>
class Cat: public Op<T> {
public:
    void forward() override {}
    void backward() override {};
    void createGradGraph() override {};
    std::string name() override {};
};

}

#endif