#ifndef OP_DIV_N_H
#define OP_DIV_N_H

#include "auto_engine/op/reshape.h"
#include "auto_engine/op/sum_expand.h"
#include "bop.h"
#include "sub.h"
#include "mul.h"
#include "data_op.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class DivN: public BOP<T> {
public:
    DivN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) override {
        return div_n(arg1, arg2);
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return div_n(grad, arg2);
        } else {
            return reshape(sum(zero(arg1) - grad * div_n(this->getOutput(), arg2)), shape<T, SHAPE>(arg2));
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return std::make_shared<DivN<T, SHAPE>>(grad, arg2);
        } else {
            auto item1 = std::make_shared<Mul<T>>(arg2, arg2);
            auto item2 = std::make_shared<DivN<T, SHAPE>>(arg1, item1);
            auto item3 = std::make_shared<Mul<T>>(grad, item2);
            auto item4 = std::make_shared<Sub<T>>(std::make_shared<DataOp<T>>(zero<T>(arg1->template getOutput())), item3);
            auto item5 = std::make_shared<Sum<T, SHAPE>>(item4);
            return std::make_shared<Reshape<T, SHAPE>>(item5, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }

    std::string name() const override {return "DivN";}
};

}
#endif