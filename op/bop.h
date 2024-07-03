#ifndef OP_BOP_H
#define OP_BOP_H

#include "op.h"
#include "data_op.h"
#include "methods.h"
#include <memory>

namespace op {

// 双变量算子（N变量可以分拆成多个双变量算子，双变量算子对计算优化更友好）

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
                this->arg1()->template setGrad(grad * this->deriv(0, arg1, arg2));
            } else {
                this->arg1()->template setGrad(this->arg1()->template getGrad() + grad * this->deriv(0, arg1, arg2));
            }
        }
        if (this->arg2()->template getRequiresGrad()) {
            if (!this->arg2()->template hasGrad()) {
                this->arg2()->template setGrad(grad * this->deriv(1, arg1, arg2));
            } else {
                this->arg2()->template setGrad(this->arg2()->template getGrad() + grad * this->deriv(1, arg1, arg2));
            }
        }
    }

    void createGradGraph() override;
};

// 乘/加会被作为基础依赖，必须声明在这个文件中

template<typename T>
class Mul: public BOP<T> {
public:
    Mul(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): op::BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) {
        return arg1 * arg2;
    }
    T deriv(u32 index, const T& arg1, const T& arg2) {
        if (index == 0) {
            return arg2;
        } else {
            return arg1;
        }
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        if (index == 0) {
            return arg2;
        } else {
            return arg1;
        }
    }
    std::string name() {return "Mul";}
};

template<typename T>
class Add: public BOP<T> {
public:
    Add(std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2): op::BOP<T>(arg1, arg2) {}

    T call(const T& arg1, const T& arg2) {
        return arg1 + arg2;
    }
    T deriv(u32 index, const T& arg1, const T& arg2) {
        return one<T>(arg1);
    }
    std::shared_ptr<Op<T>> derivFunc(u32 index, std::shared_ptr<Op<T>> arg1, std::shared_ptr<Op<T>> arg2) {
        return std::make_shared<DataOp<T>>(one<T>(arg1->template getOutput()));
    }
    std::string name() {return "Add";}
};

// bop createGradGraph方法实现
template<typename T>
void BOP<T>::createGradGraph() {
    if (!this->getRequiresGrad()) {
        return;
    }
    auto grad_graph = this->getGradGraph();
    if (this->arg1()->template getRequiresGrad()) {
        auto item = std::make_shared<Mul<T>>(grad_graph, this->derivFunc(0, this->arg1(), this->arg2()));
        if (!this->arg1()->template hasGradGraph()) {
            this->arg1()->template setGradGraph(item);
        } else {
            this->arg1()->template setGradGraph(std::make_shared<Add<T>>(this->arg1()->template getGradGraph(), item));
        }
    }
    if (this->arg2()->template getRequiresGrad()) {
        auto item = std::make_shared<Mul<T>>(grad_graph, this->derivFunc(1, this->arg1(), this->arg2()));
        if (!this->arg2()->template hasGradGraph()) {
            this->arg2()->template setGradGraph(item);
        } else {
            this->arg2()->template setGradGraph(std::make_shared<Add<T>>(this->arg2()->template getGradGraph(), item));
        }
    }
}

}

#endif