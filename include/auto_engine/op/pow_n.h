#ifndef OP_POW_N_H
#define OP_POW_N_H

#include "auto_engine/op/mul_n.h"
#include "auto_engine/op/sum_expand.h"
#include "bop.h"
#include "data_op.h"
#include "log.h"
#include "sub.h"
#include "methods.h"
#include <memory>
namespace op {

template<typename T, typename SHAPE>
class PowN: public BOP<T> {
protected:
    PowN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        auto op = std::shared_ptr<PowN<T, SHAPE>>(new PowN<T, SHAPE>(arg1, arg2));
        op->template forward();
        return op;
    }

    T call(const T& arg1, const T& arg2) override {
        return pow_n(arg1, arg2);
    }
    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad * mul_n(this->getOutput() / arg1, arg2);
        } else {
            return reshape(sum(grad * log(arg1) * this->getOutput()), shape<T, SHAPE>(arg2));
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            auto item1 = Sub<T>::op(arg2, DataOp<T>::op(one<T>(arg2->template getOutput())));
            auto item2 = PowN<T, SHAPE>::op(arg1, item1);
            auto item3 = MulN<T, SHAPE>::op(item2, arg2);
            return Mul<T>::op(grad, item3);
        } else {
            auto item1 = Log<T>::op(arg1);
            auto item2 = PowN<T, SHAPE>::op(arg1, arg2);
            auto item3 = Mul<T>::op(item1, item2); 
            auto item4 = Mul<T>::op(grad, item3);
            auto item5 = Sum<T, SHAPE>::op(item4);
            return Reshape<T, SHAPE>::op(item5, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }

    std::string name() const override {return "PowN";}
}; 

}

#endif