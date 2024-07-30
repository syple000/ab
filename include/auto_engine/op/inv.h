#ifndef OP_INV_H
#define OP_INV_H

#include "uop.h"
#include "data_op.h"
#include "methods.h"
#include "transpose.h"
#include "mmul.h"
#include "sub.h"
#include <memory>

namespace op {

template<typename T>
class Inv: public UOP<T> {
public:
    Inv(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return inv(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        auto item1 = transpose(this->getOutput(), -2, -1);
        auto item2 = mmul(item1, grad);
        auto item3 = mmul(item2, item1);
        return zero(arg) - item3; 
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        auto item1 = std::make_shared<Inv<T>>(arg);
        auto item2 = std::make_shared<Transpose<T>>(item1, -2, -1);
        auto item3 = std::make_shared<Mmul<T>>(std::make_shared<Mmul<T>>(item2, grad), item2);
        return std::make_shared<Sub<T>>(std::make_shared<DataOp<T>>(zero(arg->template getOutput())), item3);
    }

    std::string name() const override {return "Inv";}
};

}

#endif