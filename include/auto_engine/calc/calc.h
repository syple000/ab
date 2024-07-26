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
        // 计算有幂等性，如果有结果，不进行二次计算
        if (_op->template hasOutput()) {
            return _op->template getOutput();
        }

        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            if (!a[i]->template hasOutput()) {
                a[i]->template forward();
            }
        }
        _op->template forward();
        return _op->template getOutput();
    }
    
    void deriv() {
        _op->template setGrad(op::one<T>(_op->template getOutput()));
        _op->template backward();
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template backward();
        }
    }

    void createGradGraph() {
        _op->template setGradGraph(std::make_shared<op::DataOp<T>>(op::one<T>(_op->template getOutput())));
        _op->template createGradGraph();
        const auto& a = _op->template exec_queue();
        for (int i = a.size() - 1; i >= 0; i--) {
            a[i]->template createGradGraph();
        }
    }

    void clearOutput() {
        const auto& a = _op->template exec_queue();
        for (int i = 0; i < a.size(); i++) {
            if (a[i]->template args().size() > 0) {
                a[i]->template clearOutput();
            }
        }
        _op->template clearOutput();
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