#ifndef OP_LOG_H
#define OP_LOG_H

#include "data_op.h"
#include "methods.h"
#include "uop.h"
#include "div.h"
#include <memory>

namespace op {

template<typename T>
class Log: public UOP<T> {
public:
    Log(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return log(arg);
    }

    T deriv(u32 _, const T& arg) override {
        return one<T>(arg) / arg;
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> arg) override {
        return std::make_shared<Div<T>>(std::make_shared<DataOp<T>>(one<T>(arg->template getOutput())), arg);
    }
};


}

#endif