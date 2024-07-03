#ifndef CALC_CALC_H
#define CALC_CALC_H

#include "data_op.h"
#include "methods.h"
#include "op.h"
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace calc {

template<typename T>
class Calculator {
public:
    Calculator(std::shared_ptr<op::Op<T>> op) : _op(op) {
        genGraph();
    }
    const T& call() {
        auto calls = _calls;
        auto call_dep_cnt = _call_dep_cnt;
        while (calls.size()) {
            auto bcalls = calls;
            calls.clear();
            for (auto c : bcalls) {
                LOG(INFO) << "call: " << c->template name();
                c->template forward();
                for (auto dc : _call_deped[c]) {
                    call_dep_cnt[dc] -= 1;
                    if (call_dep_cnt[dc] == 0) {
                        call_dep_cnt.erase(dc);
                        calls.insert(dc);
                    }
                }
            }
        }
        LOG(INFO) << "call done: " << call_dep_cnt.size();
        return _op->template getOutput();
    }
    
    void deriv() {
        _op->template setGrad(op::one<T>(_op->template getOutput()));
        auto derivs = _derivs;
        auto deriv_dep_cnt = _deriv_dep_cnt;
        while (derivs.size()) {
            auto bderivs = derivs;
            derivs.clear();
            for (auto d : bderivs) {
                LOG(INFO) << "deriv: " << d->template name();
                d->template backward();
                for (auto dd : _deriv_deped[d]) {
                    deriv_dep_cnt[dd] -= 1;
                    if (deriv_dep_cnt[dd] == 0) {
                        deriv_dep_cnt.erase(dd);
                        derivs.insert(dd);
                    }
                }
            }
        }
        LOG(INFO) << "deriv done: " << deriv_dep_cnt.size();
    }

    void createGradGraph() {
        _op->template setGradGraph(std::make_shared<op::DataOp<T>>(op::one<T>(_op->template getOutput())));
        auto derivs = _derivs;
        auto deriv_dep_cnt = _deriv_dep_cnt;
        while (derivs.size()) {
            auto bderivs = derivs;
            derivs.clear();
            for (auto d : bderivs) {
                LOG(INFO) << "create grad graph: " << d->template name();
                d->template createGradGraph();
                for (auto dd : _deriv_deped[d]) {
                    deriv_dep_cnt[dd] -= 1;
                    if (deriv_dep_cnt[dd] == 0) {
                        derivs.insert(dd);
                    }
                }
            }
        }
        LOG(INFO) << "create grad graph done: " << deriv_dep_cnt.size();
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

    std::unordered_map<std::shared_ptr<op::Op<T>>, std::unordered_set<std::shared_ptr<op::Op<T>>>> _call_deped;
    std::unordered_map<std::shared_ptr<op::Op<T>>, u32> _call_dep_cnt;
    std::unordered_set<std::shared_ptr<op::Op<T>>> _calls;

    std::unordered_map<std::shared_ptr<op::Op<T>>, std::unordered_set<std::shared_ptr<op::Op<T>>>> _deriv_deped;
    std::unordered_map<std::shared_ptr<op::Op<T>>, u32> _deriv_dep_cnt;
    std::unordered_set<std::shared_ptr<op::Op<T>>> _derivs;

    void genGraph() {
        std::unordered_set<std::shared_ptr<op::Op<T>>> visted;
        std::function<void(std::shared_ptr<op::Op<T>>)> recur = [&recur, &visted, this](std::shared_ptr<op::Op<T>> op) {
            visted.insert(op);
            for (auto sop : op->template args()) {
                if (_call_deped[sop].find(op) == _call_deped[sop].end()) {
                    _call_deped[sop].insert(op);
                    _call_dep_cnt[op] += 1;
                }
                if (_deriv_deped[op].find(sop) == _deriv_deped[op].end()) {
                    _deriv_deped[op].insert(sop);
                    _deriv_dep_cnt[sop] += 1;
                }
                recur(sop);
            }
        };

        recur(_op);

        for (auto op : visted) {
            if (_call_dep_cnt[op] == 0) {
                _call_dep_cnt.erase(op);
                _calls.insert(op);
            }
            if (_deriv_dep_cnt[op] == 0) {
                _deriv_dep_cnt.erase(op);
                _derivs.insert(op);
            }
        }
    }
};

}

#endif