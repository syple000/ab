#ifndef OP_SUB_H
#define OP_SUB_H

#include "bop.h"
#include "data_op.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T>
class Sub: public BOP<T> {
protected:
    Sub(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        return std::shared_ptr<Sub<T>>(new Sub<T>(arg1, arg2));
    }

    T call(const T& arg1, const T& arg2) override {
        return arg1 - arg2;
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad;
        } else {
            return zero<T>(arg1)-grad;
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return grad;
        } else {
            return Sub<T>::op(DataOp<T>::op(zero<T>(arg1->template getOutput())), grad);
        }
    }

    std::string name() const override {return "Sub";}
};

}
#endif