#ifndef SUM_D_EXPAND_D_H
#define SUM_D_EXPAND_D_H

#include "auto_engine/op/uop.h"
#include "auto_engine/op/methods.h"

namespace op {

template<typename T, typename SHAPE>
class SumD: public UOP<T> {
public:
    SumD(std::shared_ptr<Op<T>> arg, int d): UOP<T>(arg), _d(d) {}

    T call(const T& arg) override {
        return sum(arg, _d);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return expand(grad, shape<T, SHAPE>(arg), _d);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "SumD";}
private:
    int _d;
};

template<typename T, typename SHAPE>
class ExpandD: public UOP<T> {
public:
    ExpandD(std::shared_ptr<Op<T>> arg, const SHAPE& shape, int d): UOP<T>(arg), _shape(shape), _d(d) {}

    T call(const T& arg) override {
        return expand(arg, _shape, _d);
    }

    T deriv(u32 index, const T& grad, const T& arg) override {
        return sum(grad, _d);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "ExpandD";}
private:
    SHAPE _shape;
    int _d;
};

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> SumD<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<ExpandD<T, SHAPE>>(grad, shape<T, SHAPE>(arg->template getOutput()), _d);
}

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> ExpandD<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<SumD<T, SHAPE>>(grad, _d);
}

}

#endif