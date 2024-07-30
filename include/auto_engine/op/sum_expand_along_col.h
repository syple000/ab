#ifndef OP_SUM_EXPAND_ALONG_COL_H
#define OP_SUM_EXPAND_ALONG_COL_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <memory>

namespace op {

template<typename T, typename SHAPE>
class SumAlongCol: public UOP<T> {
public:
    SumAlongCol(std::shared_ptr<Op<T>> arg): UOP<T>(arg) {}

    T call(const T& arg) override {
        return sum_along_col(arg);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return expand_along_col(grad, shape<T, SHAPE>(arg));
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "SumAlongCol";}
};

template<typename T, typename SHAPE>
class ExpandAlongCol: public UOP<T> {
public:
    ExpandAlongCol(std::shared_ptr<Op<T>> arg, const SHAPE& shape): UOP<T>(arg), _shape(shape) {}

    T call(const T& arg) override {
        return expand_along_col(arg, _shape);
    }

    T deriv(u32 index, const T& grad, const T& arg) override {
        return sum_along_col(grad);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override;

    std::string name() const override {return "ExpandAlongCol";}
private:
    SHAPE _shape;
};

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> SumAlongCol<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<ExpandAlongCol<T, SHAPE>>(grad, shape<T, SHAPE>(arg->template getOutput()));
}

template<typename T, typename SHAPE>
std::shared_ptr<Op<T>> ExpandAlongCol<T, SHAPE>::derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) {
    return std::make_shared<SumAlongCol<T, SHAPE>>(grad);
}

}

#endif