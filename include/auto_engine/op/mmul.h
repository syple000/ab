#ifndef OP_MMUL_H
#define OP_MMUL_H

#include "bop.h"
#include "transpose.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T>
class Mmul: public BOP<T> {
protected:
    Mmul(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        auto op = std::shared_ptr<Mmul<T>>(new Mmul<T>(arg1, arg2));
        op->template forward();
        return op;
    }

    T call(const T& arg1, const T& arg2) override {
        return mmul(arg1, arg2);
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return mmul(grad, transpose(arg2, -2, -1));
        } else {
            return mmul(transpose(arg1, -2, -1), grad);
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return Mmul<T>::op(grad, Transpose<T>::op(arg2, -2, -1));
        } else {
            return Mmul<T>::op(Transpose<T>::op(arg1, -2, -1), grad);
        }
    }

    std::string name() const override {return "Mmul";}
};

}

#endif