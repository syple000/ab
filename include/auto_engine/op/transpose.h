#ifndef OP_TRANSPOSE_H
#define OP_TRANSPOSE_H

#include "methods.h"
#include "uop.h"
#include <memory>

namespace op {

template<typename T>
class Transpose: public op::UOP<T> {
public:
    Transpose(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return transpose(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return transpose(grad);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return std::make_shared<Transpose<T>>(grad);
    }

    std::string name() const override {return "Transpose";}
};

}

#endif