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
protected:
    Inv(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        auto op = std::shared_ptr<Inv<T>>(new Inv<T>(arg));
        op->template forward();
        return op;
    }

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
        auto item1 = Inv<T>::op(arg);
        auto item2 = Transpose<T>::op(item1, -2, -1);
        auto item3 = Mmul<T>::op(Mmul<T>::op(item2, grad), item2);
        return Sub<T>::op(DataOp<T>::op(zero(arg->template getOutput())), item3);
    }

    std::string name() const override {return "Inv";}
};

}

#endif