#ifndef OP_MUL_N_H
#define OP_MUL_N_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/op/reshape.h"
#include "auto_engine/op/sum_expand.h"
#include "bop.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class MulN: public BOP<T> {
public:
    MulN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): op::BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) override {
        return mulN(arg1, arg2);
    }
    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return mulN(grad, arg2);
        } else {
            return reshape(sum(grad * arg1), shape<T, SHAPE>(arg2));
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return std::make_shared<MulN<T, SHAPE>>(grad, arg2);
        } else {
            auto item1 = std::make_shared<Mul<T>>(grad, arg1);
            auto item2 = std::make_shared<Sum<T, SHAPE>>(item1);
            return std::make_shared<Reshape<T, SHAPE>>(item2, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }
    std::string name() const override {return "MulN";}
};

}

#endif