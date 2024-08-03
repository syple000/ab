#ifndef OP_CAT_SPLIT_H
#define OP_CAT_SPLIT_H

#include "auto_engine/op/add.h"
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
        std::vector<std::reference_wrapper<T>> args; args.reserve(this->template _args.size());
        for (u32 i = 0; i < this->template _args.size(); i++) {
            args.emplace_back(std::ref(this->template _args[i]));
        }
        this->template setOutput(cat(args, _d));

        std::vector<u32> sl; sl.reserve(this->template _args.size());
        for (u32 i = 0; i < this->template _args.size(); i++) {
            sl.emplace_back(this->template _args[i].shape().getDim(_d));
        }
        _sl = std::move(sl);
    }
    void backward() override {
        auto grad = this->template getGrad();
        auto v = split(grad, _sl, _d);
        for (u32 i = 0; i < this->template _args.size(); i++) {
            auto arg = this->template _args[i];
            auto val = v[i];
            if (arg->template hasGrad()) {
                arg->template setGrad(arg->template getGrad() + val);
            } else {
                arg->template setGrad(val);
            }
        }
    }
    void createGradGraph() override {
        auto grad_graph = this->template getGradGraph();
        auto v = Split<T>::op(grad_graph, _sl, _d);
        for (u32 i = 0; i < this->template _args.size(); i++) {
            auto arg = this->template _args[i];
            auto val = v[i];
            if (arg->template getGradGraph()) {
                arg->template setGradGraph(Add<T>::op(arg->template getGradGraph(), val));
            } else {
                arg->template setGradGraph(val);
            }
        }
    }
    std::string name() override {return "Cat";}
private:
    int _d;
    std::vector<u32> _sl;
};

template<typename T>
class Split: public Op<T> {
protected:
    Split(std::shared_ptr<Op<T>> arg, const std::vector<u32>& sl, int d): Op<T>(arg), _sl(sl), _d(d) {
        for (u32 i = 0; i < sl.size(); i++) {
            this->template _outputs.emplace_back(LinkOp<T>::op(this->template shared_from_this()));
        }
    }
public:
    static const std::vector<std::shared_ptr<Op<T>>>& op(std::shared_ptr<Op<T>> arg, const std::vector<u32>& sl, int d) {
        auto op = std::shared_ptr<Op<T>>(new Split(arg, sl, d));
        return op->template getOutputs();
    }

    void forward() override {
        auto arg = this->template arg()->getOutput();
        auto v = split(arg, _sl, _d);
        for (u32 i = 0; i < this->template _outputs.size(); i++) {
            this->template _outputs[i]->template setOutput(v[i]);
        }
    }
    void backward() override {
        std::vector<std::reference_wrapper<T>> outputs; outputs.reserve(this->template _outputs.size());
        for (u32 i = 0; i < this->template _outputs.size(); i++) {
            outputs.emplace_back(std::ref(this->template _outputsp[i]->getGrad()));
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
        std::vector<std::shared_ptr<Op<T>>> v;
        for (u32 i = 0; i < this->template _outputs.size(); i++) {
            v.emplace_back(this->template _outputs[i]->getGradGraph());
        }
        auto grad_graph = op::Cat<T>::op(v, _d);
        auto arg = this->template arg();
        if (arg->template getGradGraph()) {
            arg->template setGradGraph(Add<T>::op(arg->template getGradGraph(), grad_graph));
        } else {
            arg->template setGradGraph(grad_graph);
        }
    }
    std::string name() override {return "Split";}
private:
    int _d;
    std::vector<u32> _sl;
};

}

#endif