#ifndef OP_DIV_H
#define OP_DIV_H

#include "bop.h"
#include "sub.h"
#include "mul.h"
#include "data_op.h"
#include "methods.h"
#include <memory>

namespace op {

template<typename T>
class Div: public BOP<T> {
public:
    Div(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) override {
        return arg1 / arg2;
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad / arg2;
        } else {
            return zero<T>(arg1) - grad * this->getOutput() / arg2;
        }
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            return std::make_shared<Div<T>>(grad, arg2);
        } else {
            auto item1 = std::make_shared<Mul<T>>(arg2, arg2);
            auto item2 = std::make_shared<Div<T>>(arg1, item1);
            auto item3 = std::make_shared<Mul<T>>(grad, item2);
            return std::make_shared<Sub<T>>(std::make_shared<DataOp<T>>(zero<T>(arg1->template getOutput())), item3);
        }
    }

    std::string name() const override {return "Div";}
};

}
#endif