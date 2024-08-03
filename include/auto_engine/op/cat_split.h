#ifndef OP_CAT_SPLIT_H
#define OP_CAT_SPLIT_H

#include "auto_engine/op/op.h"

namespace op {

template<typename T>
class Cat: public Op<T> {
public:
    void forward() override {}
    void backward() override {};
    void createGradGraph() override {};
    std::string name() override {};
};

}

#endif