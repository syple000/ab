#ifndef CALC_CALC_H
#define CALC_CALC_H

#include "data_op.h"
#include "methods.h"
#include "op.h"
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace calc {

template<typename T>
class Calculator {
public:
    Calculator(std::shared_ptr<op::Op<T>> op) : _op(op) {
        _pipe = getPipe(); // 按正向传播的执行顺序排列
    }
    const T& call() {
        for (auto op : _pipe) {
            op->template forward();
        }
        return _op->template getOutput();
    }
    
    void deriv() {
        _op->template setGrad(op::one<T>(_op->template getOutput()));
        for (int i = _pipe.size() - 1; i >= 0; i--) {
            _pipe[i]->template backward();
        }
    }

    void createGradGraph() {
        _op->template setGradGraph(std::make_shared<op::DataOp<T>>(op::one<T>(_op->template getOutput())));
        for (int i = _pipe.size() - 1; i >= 0; i--) {
            _pipe[i]->template createGradGraph();
        }
    }

    void clearGrad() {
        std::unordered_set<op::Op<T>*> visted;
        std::function<void(std::shared_ptr<op::Op<T>>)> recur = [&recur, &visted](std::shared_ptr<op::Op<T>> op) {
            if (visted.find(op.get()) != visted.end()) {
                return;
            }
            visted.insert(op.get());
            for (auto sop : op->template args()) {
                recur(sop);
            }
            op->template clearGrad();
        };
        recur(_op);
    }

    void clearGradGraph() {
        std::unordered_set<op::Op<T>*> visted;
        std::function<void(std::shared_ptr<op::Op<T>>)> recur = [&recur, &visted](std::shared_ptr<op::Op<T>> op) {
            if (visted.find(op.get()) != visted.end()) {
                return;
            }
            visted.insert(op.get());
            for (auto sop : op->template args()) {
                recur(sop);
            }
            op->template clearGradGraph();
        };
        recur(_op);
    }


private:
    std::shared_ptr<op::Op<T>> _op;
    std::vector<std::shared_ptr<op::Op<T>>> _pipe;

    std::vector<std::shared_ptr<op::Op<T>>> getPipe() {
        std::vector<std::shared_ptr<op::Op<T>>> pipe;
        std::unordered_set<op::Op<T>*> visted;
        std::function<void(std::shared_ptr<op::Op<T>>)> recur = [&recur, &pipe, &visted](std::shared_ptr<op::Op<T>> op) {
            if (visted.find(op.get()) != visted.end()) {
                return;
            }
            visted.insert(op.get());
            for (auto sop : op->template args()) {
                recur(sop);
            }
            pipe.push_back(op);
        };
        recur(_op);
        return pipe;
    }
};

}

#endif