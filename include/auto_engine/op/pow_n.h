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
public:
    PowN(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

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
            auto item1 = std::make_shared<Sub<T>>(arg2, std::make_shared<DataOp<T>>(one<T>(arg2->template getOutput())));
            auto item2 = std::make_shared<PowN<T, SHAPE>>(arg1, item1);
            auto item3 = std::make_shared<MulN<T, SHAPE>>(item2, arg2);
            return std::make_shared<Mul<T>>(grad, item3);
        } else {
            auto item1 = std::make_shared<Log<T>>(arg1);
            auto item2 = std::make_shared<PowN<T, SHAPE>>(arg1, arg2);
            auto item3 = std::make_shared<Mul<T>>(item1, item2); 
            auto item4 = std::make_shared<Mul<T>>(grad, item3);
            auto item5 = std::make_shared<Sum<T, SHAPE>>(item4);
            return std::make_shared<Reshape<T, SHAPE>>(item5, shape<T, SHAPE>(arg2->template getOutput()));
        }
    }

    std::string name() const override {return "PowN";}
}; 

}

#endif