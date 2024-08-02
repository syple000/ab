#ifndef OP_BOP_H
#define OP_BOP_H

#include "op.h"
#include <memory>

namespace op {

// 双变量算子（N变量可以分拆成多个双变量算子，双变量算子对计算优化更友好）

template<typename T>
class Add; // 计算图需要对grad进行加和

template<typename T>
class BOP: public Op<T>, public GenOpFuncT<2, T> {
public:
    BOP(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): Op<T>(arg1, arg2) {}

    void forward() override {
        auto arg1 = this->arg1()->template getOutput();
        auto arg2 = this->arg2()->template getOutput();
        this->setOutput(std::move(this->call(arg1, arg2)));
    }

    void backward() override {
        if (!this->getRequiresGrad()) {
            return;
        }
        auto arg1 = this->arg1()->template getOutput();
        auto arg2 = this->arg2()->template getOutput();
        auto grad = this->getGrad();
        if (this->arg1()->template getRequiresGrad()) {
            if (!this->arg1()->template hasGrad()) {
                this->arg1()->template setGrad(std::move(this->deriv(0, grad, arg1, arg2)));
            } else {
                this->arg1()->template setGrad(this->arg1()->template getGrad() + this->deriv(0, grad, arg1, arg2));
            }
        }
        if (this->arg2()->template getRequiresGrad()) {
            if (!this->arg2()->template hasGrad()) {
                this->arg2()->template setGrad(std::move(this->deriv(1, grad, arg1, arg2)));
            } else {
                this->arg2()->template setGrad(this->arg2()->template getGrad() + this->deriv(1, grad, arg1, arg2));
            }
        }
    }

    void createGradGraph() override {
        if (!this->getRequiresGrad()) {
            return;
        }
        auto grad_graph = this->getGradGraph();
        if (this->arg1()->template getRequiresGrad()) {
            auto item = this->derivFunc(0, grad_graph, this->arg1(), this->arg2());
            if (!this->arg1()->template hasGradGraph()) {
                this->arg1()->template setGradGraph(item);
            } else {
                this->arg1()->template setGradGraph(Add<T>::op(this->arg1()->template getGradGraph(), item));
            }
        }
        if (this->arg2()->template getRequiresGrad()) {
            auto item = this->derivFunc(1, grad_graph, this->arg1(), this->arg2());
            if (!this->arg2()->template hasGradGraph()) {
                this->arg2()->template setGradGraph(item);
            } else {
                this->arg2()->template setGradGraph(Add<T>::op(this->arg2()->template getGradGraph(), item));
            }
        }
    }

};

}

#endif