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

    const T& call() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            // 计算有幂等性，如果有结果，不进行二次计算
            if (!a[i]->template hasOutput()) {
                a[i]->template forward();
            }
        }
        return _op->template getOutput();
    }
    
    void deriv() {
        _op->template setGrad(op::one<T>(_op->template getOutput()));
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template backward();
        }
    }

    void createGradGraph() {
        _op->template setGradGraph(std::make_shared<op::DataOp<T>>(op::one<T>(_op->template getOutput())));
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template createGradGraph();
        }
    }

    void clearOutput() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            a[i]->template clearOutput();
        }
    }

    void clearGrad() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            a[i]->template clearGrad();
        }
    }

    void clearGradGraph() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            a[i]->template clearGradGraph();
        }
    }


private:
    std::shared_ptr<op::Op<T>> _op;
};

}

#endif