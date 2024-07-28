#ifndef OP_SUB_N_H
#define OP_SUB_N_H

#include "auto_engine/op/bop.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/methods.h"
#include "auto_engine/op/reshape.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/sum_expand.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class SubN: public BOP<T> {
public:
    SubN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) override {
        return subN(arg1, arg2);
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad;
        } else {
            return reshape(sum(zero(arg1) - grad), shape<T, SHAPE>(arg2));
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return grad;
        } else {
            auto zeros = std::make_shared<DataOp<T>>(zero(arg1->template getOutput()));
            auto neg_grad = std::make_shared<Sub<T>>(zeros, grad);
            auto item = std::make_shared<Sum<T, SHAPE>>(neg_grad);
            return std::make_shared<Reshape<T, SHAPE>>(item, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }

    std::string name() const override {return "SubN";}
};



}

#endif