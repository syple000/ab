#ifndef ALGO_OPT_H
#define ALGO_OPT_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace algo {

class Optimizer {
public:
    Optimizer(std::function<std::shared_ptr<op::Op<base::Tensor<f64>>>(const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>&)> cost_func,
        const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>& vars) : _cost_func(cost_func), _vars(vars) {}
    
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
    std::function<std::shared_ptr<op::Op<base::Tensor<f64>>>(const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>&)> _cost_func;
    std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>> _vars;
    
    std::string _algo;
    std::unordered_map<std::string, f64> _hyper_params;

private: // 参数获取方法
    inline f64 getHyperParam(const std::string& param_name, f64 default_val) const {
        auto iter = _hyper_params.find(param_name);
        if (iter == _hyper_params.end()) {
            return default_val;
        }
        return iter->second;
    }
    inline f64 mustGetHyperParam(const std::string& param_name) const {
        auto iter = _hyper_params.find(param_name);
        if (iter == _hyper_params.end()) {
            throw std::runtime_error(fmt::format("hyper param not found: {}", param_name));
        }
        return iter->second;
    }

private: // 通用方法
    f64 calcCost(std::shared_ptr<op::Op<base::Tensor<f64>>> cost) {
        calc::Calculator<base::Tensor<f64>> c(cost);
        c.call();
        const auto& t = cost->getOutput();
        if (t.shape().tensorSize() != 1) {
            throw std::runtime_error("cost tensor size != 1");
        }
        return t.data()[0];
    }

    void calcGrad(std::shared_ptr<op::Op<base::Tensor<f64>>> cost) {
        calc::Calculator<base::Tensor<f64>> c(cost);
        c.clearGrad();
        c.deriv();
    }

    f64 grad2rdNorm() {
        f64 res = 0;
        for (int i = 0; i < _vars.size(); i++) {
            auto grad = _vars[i]->getGrad();
            auto grad_pow2 = grad * grad;
            res += grad_pow2.sum().data()[0];
        }
        return res;
    }
    f64 grad2rdNorm(const std::vector<base::Tensor<f64>>& grads) {
        f64 res = 0;
        for (int i = 0; i < grads.size(); i++) {
            auto grad_pow2 = grads[i] * grads[i];
            res += grad_pow2.sum().data()[0];
        }
        return res;
    }

    void updateVars(const std::vector<base::Tensor<f64>>& vars, f64 step) {
        for (int i = 0; i < _vars.size(); i++) {
            _vars[i]->setOutput(vars[i] - base::Tensor<f64>(vars[i].shape(), step) * _vars[i]->getGrad());
        }
    }
    void updateVars(f64 step, const std::vector<base::Tensor<f64>>& direct) {
        for (int i = 0; i < _vars.size(); i++) {
            _vars[i]->setOutput(_vars[i]->getOutput() - base::Tensor<f64>(_vars[i]->getOutput().shape(), step) * direct[i]);
        }
    }

    std::vector<base::Tensor<f64>> getVars() { // 产生一次拷贝
        std::vector<base::Tensor<f64>> vals;
        vals.reserve(_vars.size());
        for (int i = 0; i < _vars.size(); i++) {
            vals.emplace_back(_vars[i]->getOutput());
        }
        return vals;
    }

    void restoreVars(std::vector<base::Tensor<f64>>&& vals) {
        for (int i = 0; i < _vars.size(); i++) {
            _vars[i]->setOutput(std::move(vals[i]));
        }
    }

    std::string gradStr() {
        std::stringstream stream;
        for (int i = 0; i < _vars.size(); i++) {
            if (i != 0) {stream << " ";}
            stream << i << "th: " << _vars[i]->getGrad().toString(true);
        }
        return stream.str();
    }

    std::string gradStr(const std::vector<base::Tensor<f64>>& grads) {
        std::stringstream stream;
        for (int i = 0; i < grads.size(); i++) {
            if (i != 0) {stream << " ";}
            stream << i << "th: " << grads[i].toString(true);
        }
        return stream.str();
    }

private: // 终止检查
    bool checkGrad2rdNorm(f64 norm) {
        if (norm < 0) {
            LOG(ERROR) << "fatal: norm < 0";
            throw std::runtime_error("norm lt 0");
        }
        if (getHyperParam("enable_check_grad2rd_norm", 1) < 0) {
            return true;
        }
        auto sqrt_norm = sqrt(norm);
        if (getHyperParam("grad2rd_norm_threshold", EPSILON) > sqrt_norm) {
            LOG(INFO) << "check norm lt threshold";
            return false;
        }
        return true;
    }

    u32 _insign_cost_diff_cnt = 0;
    bool checkCostDiff(f64 pre, f64 cur) {
        if (getHyperParam("enable_check_cost_diff", 1) < 0) {
            return true;
        }
        if (getHyperParam("check_cost_diff_threshold", EPSILON) > 2 * abs(cur - pre) / (abs(pre + cur) + EPSILON)) {
            LOG(INFO) << "check cost diff insignificant, cnt++";
            _insign_cost_diff_cnt += 1;
        } else {
            _insign_cost_diff_cnt = 0;
        }
        if ((u32)round(getHyperParam("cost_diff_patient", 3)) <= _insign_cost_diff_cnt) {
            LOG(INFO) << "check cost diff, cnt reach patient";
            return false;
        }
        return true;
    }

private: // 算法实现
    void gradDescent();
    void adam();
};

}

#endif