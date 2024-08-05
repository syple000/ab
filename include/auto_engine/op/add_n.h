#ifndef OP_ADD_N_H
#define OP_ADD_N_H

#include "auto_engine/op/bop.h"
#include "auto_engine/op/methods.h"
#include "auto_engine/op/reshape.h"
#include "auto_engine/op/sum_expand.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class AddN: public BOP<T> {
protected:
    AddN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        auto op = std::shared_ptr<AddN<T, SHAPE>>(new AddN<T, SHAPE>(arg1, arg2));
        op->template forward();
        return op;
    }

    T call(const T& arg1, const T& arg2) override {
        return add_n(arg1, arg2);
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad;
        } else {
            return reshape(sum(grad), shape<T, SHAPE>(arg2));
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return grad;
        } else {
            auto item = Sum<T, SHAPE>::op(grad);
            return Reshape<T, SHAPE>::op(item, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }

    std::string name() const override {return "AddN";}
};



}

#endif