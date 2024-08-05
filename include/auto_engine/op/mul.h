#ifndef OP_MUL_H
#define OP_MUL_H

#include "bop.h"
#include <memory>

namespace op {

template<typename T>
class Mul: public BOP<T> {
protected:
    Mul(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): op::BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        auto op = std::shared_ptr<Mul<T>>(new Mul<T>(arg1, arg2));
        op->template forward();
        return op;
    }

    T call(const T& arg1, const T& arg2) override {
        return arg1 * arg2;
    }
    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad * arg2;
        } else {
            return grad * arg1;
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return Mul<T>::op(grad, arg2);
        } else {
            return Mul<T>::op(grad, arg1);
        }
    }
    std::string name() const override {return "Mul";}
};

}

#endif