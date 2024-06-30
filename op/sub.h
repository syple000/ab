#ifndef OP_SUB_H
#define OP_SUB_H

#include "bop.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T>
class Sub: public BOP<T> {
public:
    Sub(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) {
        return arg1 - arg2;
    }

    T deriv(u32 index, const T& arg1, const T& arg2) {
        if (index == 0) {
            return one<T>(arg1);
        } else {
            return zero<T>(arg1)-one<T>(arg1);
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        if (index == 0) {
            return std::make_shared<DataOp<T>>(one<T>(arg1->template getOutput()));
        } else {
            return std::make_shared<DataOp<T>>(zero<T>(arg1->template getOutput())-one<T>(arg1->template getOutput()));
        }
    }
};

}
#endif