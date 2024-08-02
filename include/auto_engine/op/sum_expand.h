#ifndef OP_SUM_EXPAND_H
#define OP_SUM_EXPAND_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class Sum: public UOP<T> {
protected:
    Sum(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        return std::shared_ptr<Sum<T, SHAPE>>(new Sum<T, SHAPE>(arg));
    }

    T call(const T& arg) override {
        return sum(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return expand(grad, shape<T, SHAPE>(arg));
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "Sum";}
};

template<typename T, typename SHAPE>
class Expand: public UOP<T> {
protected:
    Expand(std::shared_ptr<Op<T>> arg, const SHAPE& shape): UOP<T>(arg), _shape(shape) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg, const SHAPE& shape) {
        return std::shared_ptr<Expand<T, SHAPE>>(new Expand<T, SHAPE>(arg, shape));
    }

    T call(const T& arg) override {
        return expand(arg, _shape);
    }

    T deriv(u32 index, const T& grad, const T& arg) override {
        return sum(grad);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "Expand";}
private:
    SHAPE _shape;
};

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> Sum<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return Expand<T, SHAPE>::op(grad, shape<T, SHAPE>(arg->template getOutput()));
}

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> Expand<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return Sum<T, SHAPE>::op(grad);
}

}

#endif