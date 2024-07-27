#ifndef ALGO_ALGO_H
#define ALGO_ALGO_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace algo {

class OptAlgo {
public:
    OptAlgo(std::shared_ptr<op::Op<base::Tensor<f64>>> cost, const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>> vars) : _cost(cost), _vars(vars) {}
    
    void algoHyperParams(const std::string& algo, const std::unordered_map<std::string, f64>& hyper_params) {
        _algo = algo;
        _hyper_params = hyper_params;
    }

    void run() {
        if (_algo == "grad_descent") {
            gradDescent();
        } else if (_algo == "adam") {
            adam();
        } else {
            throw std::runtime_error(fmt::format("{} invalid", _algo));
        }
    }

private:
    std::shared_ptr<op::Op<base::Tensor<f64>>> _cost;
    std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>> _vars;
    
    std::string _algo;
    std::unordered_map<std::string, f64> _hyper_params;
    inline f64 getHyperParam(const std::string& param_name, f64 default_val) {
        auto iter = _hyper_params.find(param_name);
        if (iter == _hyper_params.end()) {
            return default_val;
        }
        return iter->second;
    }
    inline f64 mustGetHyperParam(const std::string& param_name) {
        auto iter = _hyper_params.find(param_name);
        if (iter == _hyper_params.end()) {
            throw std::runtime_error(fmt::format("hyper param not found: {}", param_name));
        }
        return iter->second;
    }

    f64 calc_cost() {
        calc::Calculator<base::Tensor<f64>> c(_cost);
        c.clearOutput();
        c.call();
        const auto& cost = _cost->getOutput();
        if (cost.shape().tensorSize() != 1) {
            throw std::runtime_error("cost tensor size != 1");
        }
        return cost.data()[0];
    }

    void calc_grad() {
        calc::Calculator<base::Tensor<f64>> c(_cost);
        if (!_cost->hasOutput()) {calc_cost();}
        c.clearGrad();
        c.deriv();
    }

    void gradDescent();
    void adam() {}
};

}

#endif