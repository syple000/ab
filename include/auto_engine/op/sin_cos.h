#ifndef OP_SIN_H
#define OP_SIN_H

#include "data_op.h"
#include "sub.h"
#include "mul.h"
#include "methods.h"
#include "uop.h"
#include <cmath>
#include <memory>

namespace op {

template<typename T>
class Sin: public UOP<T> {
protected:
    Sin(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        return std::shared_ptr<Sin<T>>(new Sin<T>(arg));
    }

    T call(const T& arg) override {
        return sin(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return grad * cos(arg);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "Sin";}
};

template<typename T>
class Cos: public UOP<T> {
protected:
    Cos(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        return std::shared_ptr<Cos<T>>(new Cos<T>(arg));
    }

    T call(const T& arg) override {
        return cos(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return zero<T>(arg) - grad *sin(arg);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "Cos";}
};

template<typename T>
std::shared_ptr<Op<T>> Sin<T>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    auto item = Cos<T>::op(arg);
    return Mul<T>::op(grad, item);
}

template<typename T>
std::shared_ptr<Op<T>> Cos<T>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    auto item = Sub<T>::op(
        DataOp<T>::op(zero<T>(arg->template getOutput())),
        Sin<T>::op(arg)
    );
    return Mul<T>::op(grad, item);
}

}

#endif