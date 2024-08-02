#ifndef OP_UOP_H
#define OP_UOP_H

#include "op.h"
#include "bop.h"
#include <memory>

namespace op {

// 单变量算子

template<typename T>
class UOP: public Op<T>, public GenOpFuncT<1, T> {
public:
    UOP(std::shared_ptr<Op<T>> arg): Op<T>(arg) {}

    void forward() override {
        auto arg = this->arg()->template getOutput();
        this->setOutput(std::move(this->call(arg)));
    }

    void backward() override {
        if (!this->getRequiresGrad()) {
            return;
        }
        auto grad = this->getGrad();
        auto arg = this->arg()->template getOutput();
        if (!this->arg()->template hasGrad()) {
            this->arg()->template setGrad(std::move(this->deriv(0, grad, arg)));
        } else {
            this->arg()->template setGrad(this->arg()->template getGrad() + this->deriv(0, grad, arg));
        }
    }

    void createGradGraph() override {
        if (!this->getRequiresGrad()) {
            return;
        }
        auto grad_graph = this->getGradGraph();
        auto item = this->derivFunc(0, grad_graph, this->arg());
        if (!this->arg()->template hasGradGraph()) {
            this->arg()->template setGradGraph(item);
        } else {
            this->arg()->template setGradGraph(Add<T>::op(this->arg()->template getGradGraph(), item));
        }
    }
};

}

#endif