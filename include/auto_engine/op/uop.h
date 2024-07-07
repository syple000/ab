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
            this->arg()->template setGrad(grad * this->deriv(0, arg));
        } else {
            this->arg()->template setGrad(this->arg()->template getGrad() + grad * this->deriv(0, arg));
        }
    }

    void createGradGraph() override {
        if (!this->getRequiresGrad()) {
            return;
        }
        auto grad_graph = this->getGradGraph();
        auto item = std::make_shared<Mul<T>>(grad_graph, this->derivFunc(0, this->arg()));
        if (!this->arg()->template hasGradGraph()) {
            this->arg()->template setGradGraph(item);
        } else {
            this->arg()->template setGradGraph(std::make_shared<Add<T>>(this->arg()->template getGradGraph(), item));
        }
    }
};

}

#endif