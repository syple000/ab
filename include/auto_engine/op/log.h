#ifndef OP_LOG_H
#define OP_LOG_H

#include "methods.h"
#include "uop.h"
#include "div.h"
#include <memory>

namespace op {

template<typename T>
class Log: public UOP<T> {
protected:
    Log(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        auto op = std::shared_ptr<Log<T>>(new Log<T>(arg));
        op->template forward();
        return op;
    }

    T call(const T& arg) override {
        return log(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return grad / arg;
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return Div<T>::op(grad, arg);
    }

    std::string name() const override {return "Log";}
};


}

#endif