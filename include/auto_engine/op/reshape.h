#ifndef OP_RESHAPE_H
#define OP_RESHAPE_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <cstdlib>
#include <memory>
namespace op {

template<typename T, typename SHAPE>
class Reshape: public UOP<T> {
protected:
    Reshape(std::shared_ptr<Op<T>> arg, const SHAPE& shape): UOP<T>(arg), _shape(shape) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg, const SHAPE& shape) {
        return std::shared_ptr<Reshape<T, SHAPE>>(new Reshape<T, SHAPE>(arg, shape));
    }

    T call(const T& arg) override {
        return reshape(arg, _shape);
    }

    T deriv(u32 index, const T& grad, const T& arg) override {
        return reshape(grad, shape<T, SHAPE>(arg));
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return Reshape<T, SHAPE>::op(grad, shape<T, SHAPE>(arg->template getOutput()));
    }

    std::string name() const override {return "Reshape";}
private:
    SHAPE _shape;
};

}

#endif