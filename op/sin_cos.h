#ifndef OP_SIN_H
#define OP_SIN_H

#include "basic_types.h"
#include "data_op.h"
#include "sub.h"
#include "methods.h"
#include "uop.h"
#include <cmath>
#include <memory>

namespace op {

template<typename T>
class Sin: public UOP<T> {
public:
    Sin(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return sin(arg);
    }

    T deriv(u32 _, const T& arg) override {
        return cos(arg);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> arg) override;
};

template<typename T>
class Cos: public UOP<T> {
public:
    Cos(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return cos(arg);
    }

    T deriv(u32 _, const T& arg) override {
        return zero<T>(arg)-sin(arg);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> arg) override;
};

template<typename T>
std::shared_ptr<Op<T>> Sin<T>::derivFunc(u32 _, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<Cos<T>>(arg);
}

template<typename T>
std::shared_ptr<Op<T>> Cos<T>::derivFunc(u32 _, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<Sub<T>>(
        std::make_shared<DataOp<T>>(zero<T>(arg->template getOutput())),
        std::make_shared<Sin<T>>(arg)
    );
}

}

#endif