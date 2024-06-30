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
public:
    Pow(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) {
        return pow(arg1, arg2);
    }
    T deriv(u32 index, const T& arg1, const T& arg2) {
        if (index == 0) {
            return arg2 * this->getOutput() / arg1;
        } else {
            return log(arg1) * this->getOutput();
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        if (index == 0) {
            auto item1 = std::make_shared<Sub<T>>(arg2, std::make_shared<DataOp<T>>(one<T>(arg2->template getOutput())));
            auto item2 = std::make_shared<Pow<T>>(arg1, item1);
            return std::make_shared<Mul<T>>(arg2, item2);
        } else {
            auto item1 = std::make_shared<Log<T>>(arg1);
            auto item2 = std::make_shared<Pow<T>>(arg1, arg2);
            return std::make_shared<Mul<T>>(item1, item2); 
        }
    }
}; 

}

#endif