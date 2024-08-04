#ifndef OP_CAT_SPLIT_H
#define OP_CAT_SPLIT_H

#include "auto_engine/op/add.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/link_op.h"
#include "auto_engine/op/methods.h"
#include "auto_engine/op/op.h"
#include <functional>
#include <memory>
#include <vector>

namespace op {

template<typename T>
class Split;

template<typename T>
class Cat: public Op<T> {
protected:
    Cat(const std::vector<std::shared_ptr<Op<T>>>& args, int d): Op<T>(args), _d(d) {}
public:
    static std::shared_ptr<Op<T>> op(const std::vector<std::shared_ptr<Op<T>>>& args, int d) {
        return std::shared_ptr<Op<T>>(new Cat(args, d));
    }

    void forward() override {
        std::vector<std::reference_wrapper<const T>> args; args.reserve(this->template args().size());
        for (u32 i = 0; i < this->template args().size(); i++) {
            args.emplace_back(std::cref(this->template args()[i]->getOutput()));
        }
        this->template setOutput(cat(args, _d));

        std::vector<u32> sl; sl.reserve(this->template args().size());
        for (u32 i = 0; i < this->template args().size(); i++) {
            sl.emplace_back(this->template args()[i]->getOutput().shape().getDim(_d));
        }
        _sl = std::move(sl);
    }
    void backward() override {
        if (!this->template getRequiresGrad()) {return;}

        auto grad = this->template getGrad();
        auto v = split(grad, _sl, _d);
        for (u32 i = 0; i < this->template args().size(); i++) {
            auto arg = this->template args()[i];
            if (!arg->template getRequiresGrad()) {continue;}

            auto val = v[i];
            if (arg->template hasGrad()) {
                arg->template setGrad(arg->template getGrad() + val);
            } else {
                arg->template setGrad(val);
            }
        }
    }
    void createGradGraph() override {
        if (!this->template getRequiresGrad()) {return;}

        auto grad_graph = this->template getGradGraph();
        auto v = Split<T>::op(grad_graph, _sl, _d);
        for (u32 i = 0; i < this->template args().size(); i++) {
            auto arg = this->template args()[i];
            if (!arg->template getRequiresGrad()) {continue;}

            auto val = v[i];
            if (arg->template hasGradGraph()) {
                arg->template setGradGraph(Add<T>::op(arg->template getGradGraph(), val));
            } else {
                arg->template setGradGraph(val);
            }
        }
    }
    std::string name() const override {return "Cat";}
private:
    int _d;
    std::vector<u32> _sl;
};

template<typename T>
class Split: public Op<T> {
protected:
    Split(std::shared_ptr<Op<T>> arg, const std::vector<u32>& sl, int d): Op<T>(arg), _sl(sl), _d(d) {}
public:
    static std::vector<std::shared_ptr<Op<T>>> op(std::shared_ptr<Op<T>> arg, const std::vector<u32>& sl, int d) {
        auto op = std::shared_ptr<Op<T>>(new Split(arg, sl, d));
        std::vector<std::shared_ptr<Op<T>>> ops; ops.reserve(sl.size());
        std::vector<std::weak_ptr<Op<T>>> weak_ops; weak_ops.reserve(sl.size());
        for (u32 i = 0; i < sl.size(); i++) {
            ops.emplace_back(LinkOp<T>::op(op));
            weak_ops.emplace_back(ops[i]);
        }
        op->template setOutputs(weak_ops);
        return ops;
    }

    void forward() override {
        auto arg = this->template arg()->getOutput();
        _v = split(arg, _sl, _d);
        for (u32 i = 0; i < this->template getOutputs().size(); i++) {
            auto output = this->template getOutputs()[i].lock();
            if (output) {
                output->setOutput(_v[i]);
            }
        }
    }

    void backward() override {
        if (!this->template getRequiresGrad()) {return;}

        std::vector<std::reference_wrapper<const T>> outputs; outputs.reserve(this->template getOutputs().size());
        std::vector<T> v; v.reserve(this->template getOutputs().size());
        for (u32 i = 0; i < this->template getOutputs().size(); i++) {
            auto output = this->template getOutputs()[i].lock();
            if (output) {
                if (!output->hasGrad()) {
                    output->setGrad(zero(output->getOutput()));
                }
                outputs.emplace_back(std::cref(output->getGrad()));
            } else {
                v.push_back(zero(_v[i]));
                outputs.emplace_back(std::cref(v[v.size() - 1]));
            }
        }
        auto grad = cat(outputs, _d);
        auto arg = this->template arg();
        if (arg->template hasGrad()) {
            arg->template setGrad(arg->template getGrad() + grad);
        } else {
            arg->template setGrad(grad);
        }
    }
    void createGradGraph() override {
        if (!this->template getRequiresGrad()) {return;}

        std::vector<std::shared_ptr<Op<T>>> v;
        for (u32 i = 0; i < this->template getOutputs().size(); i++) {
            auto output = this->template getOutputs()[i].lock();
            if (output) {
                if (!output->hasGradGraph()) {
                    output->setGradGraph(DataOp<T>::op(zero(output->getOutput())));
                }
                v.emplace_back(output->getGradGraph());
            } else {
                auto op = DataOp<T>::op(zero(_v[i]));
                v.emplace_back(op);
            }
        }
        auto grad_graph = op::Cat<T>::op(v, _d);
        auto arg = this->template arg();
        if (arg->template hasGradGraph()) {
            arg->template setGradGraph(Add<T>::op(arg->template getGradGraph(), grad_graph));
        } else {
            arg->template setGradGraph(grad_graph);
        }
    }
    std::string name() const override {return "Split";}
private:
    int _d;
    std::vector<u32> _sl;
    // split后的结果缓存
    std::vector<T> _v;
};

}

#endif