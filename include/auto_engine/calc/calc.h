#ifndef CALC_CALC_H
#define CALC_CALC_H

#include "auto_engine/op/data_op.h"
#include "auto_engine/op/methods.h"
#include "auto_engine/op/op.h"
#include <memory>

namespace calc {

template<typename T>
class Calculator {
public:
    Calculator(std::shared_ptr<op::Op<T>> op) : _op(op) {}

    void deriv() {
        _op->template setGrad(op::one<T>(_op->template getOutput()));
        _op->template backward();
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template backward();
        }
    }

    void createGradGraph() {
        _op->template setGradGraph(op::DataOp<T>::op(op::one<T>(_op->template getOutput())));
        _op->template createGradGraph();
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template createGradGraph();
        }
    }

    void clearGrad() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            a[i]->template clearGrad();
        }
        _op->template clearGrad();
    }

    void clearGradGraph() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            a[i]->template clearGradGraph();
        }
        _op->template clearGradGraph();
    }


private:
    std::shared_ptr<op::Op<T>> _op;
};

}

#endif