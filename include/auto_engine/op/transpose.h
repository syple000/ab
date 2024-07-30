#ifndef OP_TRANSPOSE_H
#define OP_TRANSPOSE_H

#include "methods.h"
#include "uop.h"
#include <memory>

namespace op {

template<typename T>
class Transpose: public op::UOP<T> {
public:
    Transpose(std::shared_ptr<Op<T>> arg, int d1, int d2): UOP<T>(arg), _d1(d1), _d2(d2) {}

    T call(const T& arg) override {
        return transpose(arg, _d1, _d2);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return transpose(grad, _d2, _d1);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return std::make_shared<Transpose<T>>(grad, _d2, _d1);
    }

    std::string name() const override {return "Transpose";}
private:
    int _d1, _d2;
};

}

#endif