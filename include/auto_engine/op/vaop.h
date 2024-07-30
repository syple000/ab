#ifndef OP_VAOP_H
#define OP_VAOP_H

#include "op.h"
#include "bop.h"
#include <functional>
#include <memory>
#include <vector>

namespace op {

template<typename T>
class VAOP: public Op<T>, public VAOpFunc<T> {
public:
    VAOP(const std::vector<std::shared_ptr<Op<T>>>& args): Op<T>(args) {}

    void forward() override {
        std::vector<std::reference_wrapper<T>> args;
        args.reserve(this->args().size());
        for (u32 i = 0; i < this->args().size(); i++) {
            args.emplace_back(std::ref(this->args()[i]->template getOutput()));
        }
        this->setOutput(std::move(this->call(args)));
    }

    void backward() override {
        if (!this->getRequiresGrad()) {
            return;
        }

        std::vector<std::reference_wrapper<T>> args;
        args.reserve(this->args().size());
        for (u32 i = 0; i < this->args().size(); i++) {
            args.emplace_back(std::ref(this->args()[i]->template getOutput()));
        }

        for (u32 i = 0; i < this->args().size(); i++) {
            if (!this->args()[i]->template getRequiresGrad()) {
                continue;
            }
            if (!this->args()[i]->template hasGrad()) {
                this->args()[i]->template setGrad(std::move(this->deriv(i, this->getGrad(), args)));
            } else {
                this->args()[i]->template setGrad(this->args[i]->template getGrad() + this->deriv(i, this->getGrad(), args));
            }
        }
    }

    void createGradGraph() override {
        if (!this->getRequiresGrad()) {
            return;
        }

        for (u32 i = 0; i < this->args().size(); i++) {
            if (!this->args()[i]->template getRequiresGrad()) {
                continue;
            }
            auto item = this->derivFunc(i, this->getGradGraph(), this->args());
            if (!this->args()[i]->template hasGradGraph()) {
                this->args()[i]->template setGradGraph(item);
            } else {
                this->args()[i]->template setGradGraph(std::make_shared<Add<T>>(this->args()[i]->template getGradGraph(), item));
            }
        }
    }

};

}

#endif