#ifndef OP_SUM_EXPAND_ALONG_ROW_H
#define OP_SUM_EXPAND_ALONG_ROW_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class SumAlongRow: public UOP<T> {
public:
    SumAlongRow(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return sum_along_row(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return expand_along_row(grad, shape<T, SHAPE>(arg));
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "SumAlongRow";}
};

template<typename T, typename SHAPE>
class ExpandAlongRow: public UOP<T> {
public:
    ExpandAlongRow(std::shared_ptr<Op<T>> arg, const SHAPE& shape): UOP<T>(arg), _shape(shape) {}

    T call(const T& arg) override {
        return expand_along_row(arg, _shape);
    }

    T deriv(u32 index, const T& grad, const T& arg) override {
        return sum_along_row(grad);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "ExpandAlongRow";}
private:
    SHAPE _shape;
};

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> SumAlongRow<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<ExpandAlongRow<T, SHAPE>>(grad, shape<T, SHAPE>(arg->template getOutput()));
}

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> ExpandAlongRow<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<SumAlongRow<T, SHAPE>>(grad);
}

}

#endif