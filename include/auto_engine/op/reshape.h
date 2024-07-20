#ifndef OP_RESHAPE_H
#define OP_RESHAPE_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <memory>
namespace op {

template<typename T, typename SHAPE>
class Reshape: public UOP<T> {
public:
    Reshape(std::shared_ptr<Op<T>> arg, const SHAPE& shape): UOP<T>(arg), _shape(shape) {}

    T call(const T& arg) override {
        return reshape(arg, _shape);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return reshape(grad, shape(arg));
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return std::make_shared<Reshape<T, SHAPE>>(grad, shape(arg->template getOutput()));
    }

    std::string name() const override {return "Reshape";}
private:
    SHAPE _shape;
};

}

#endif