#ifndef OP_MMUL_H
#define OP_MMUL_H

#include "bop.h"
#include "transpose.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T>
class Mmul: public BOP<T> {
public:
    Mmul(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

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
            return std::make_shared<Mmul<T>>(grad, std::make_shared<Transpose<T>>(arg2, -2, -1));
        } else {
            return std::make_shared<Mmul<T>>(std::make_shared<Transpose<T>>(arg1, -2, -1), grad);
        }
    }

    std::string name() const override {return "Mmul";}
};

}

#endif