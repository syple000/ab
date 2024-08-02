#ifndef OP_ADD_H
#define OP_ADD_H

#include "bop.h"
#include <memory>

namespace op {

template<typename T>
class Add: public BOP<T> {
protected:
    Add(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): BOP<T>(arg1, arg2) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        return std::shared_ptr<Add<T>>(new Add<T>(arg1, arg2));
    }

    T call(const T& arg1, const T& arg2) override {
        return arg1 + arg2;
    }

    T deriv(u32 index, const T& grad, const T& arg1, const T& arg2) override {
        return grad;
    }

    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) override {
        return grad;
    }

    std::string name() const override {return "Add";}
};

}

#endif