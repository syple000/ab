#ifndef OP_POW_H
#define OP_POW_H

#include "bop.h"
#include "data_op.h"
#include "log.h"
#include "sub.h"
#include "methods.h"
#include <memory>
namespace op {

template<typename T>
class Pow: public BOP<T> {
protected:
    Pow(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        auto op = std::shared_ptr<Pow<T>>(new Pow<T>(arg1, arg2));
        op->template forward();
        return op;
    }

    T call(const T& arg1, const T& arg2) override {
        return pow(arg1, arg2);
    }
    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        if (index == 0) {
            return grad * arg2 * this->getOutput() / arg1;
        } else {
            return grad * log(arg1) * this->getOutput();
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        if (index == 0) {
            auto item1 = Sub<T>::op(arg2, DataOp<T>::op(one<T>(arg1->template getOutput())));
            auto item2 = Pow<T>::op(arg1, item1);
            auto item3 = Mul<T>::op(arg2, item2);
            return Mul<T>::op(grad, item3);
        } else {
            auto item1 = Log<T>::op(arg1);
            auto item2 = Pow<T>::op(arg1, arg2);
            auto item3 = Mul<T>::op(item1, item2); 
            return Mul<T>::op(grad, item3);
        }
    }

    std::string name() const override {return "Pow";}
}; 

}

#endif