#include "auto_engine/algo/opt.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <fmt/core.h>
#include <stdexcept>
#include <vector>

namespace algo {

void Optimizer::adam() {
    // 读取超参
    f64 step = mustGetHyperParam("step");
    bool enable_yogi = getHyperParam("enable_yogi", -1) > 0; // adam二次矩过大, yogi算法优化
    f64 first_coef = getHyperParam("first_coef", 0.9);
    f64 second_coef = getHyperParam("second_coef", 0.999);
    u32 max_iter_cnt = round(getHyperParam("max_iter_cnt", 100));
    LOG(INFO) << fmt::format("hyper params, step: {}, enable_yogi: {}, first_coef: {}, second_coef: {}, max_iter_cnt: {}", 
        step, enable_yogi, first_coef, second_coef, max_iter_cnt);

    if (first_coef >= 1 || first_coef <= 0 || second_coef >= 1 || second_coef <= 0) {
        LOG(ERROR) << fmt::format("adam coef invalid, first coef: {}, second coef: {}", first_coef, second_coef);
        throw std::runtime_error(fmt::format("adam coef invalid, first coef: {}, second coef: {}", first_coef, second_coef));
    }

    // 一/二阶矩向量，初始化0
    std::vector<base::Tensor<f64>> m, v, mm, vm, grads;
    m.reserve(_vars.size()); v.reserve(_vars.size()); mm.reserve(_vars.size()); vm.reserve(_vars.size()); grads.reserve(_vars.size());
    for (int i = 0; i < _vars.size(); i++) {
        m.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
        mm.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
        v.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
        vm.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
        grads.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
    }

    // 循环
    auto cost = _cost_func(_vars);
    f64 pre_cost = getCost(cost);
    u32 iter_cnt = 0;
    while (iter_cnt < max_iter_cnt) {
        calcGrad(cost);

        // 更新并校准矩向量
        for (int i = 0; i < _vars.size(); i++) {
            m[i] = m[i] * first_coef + _vars[i]->getGrad() * (1 - first_coef);
            auto grad_pow2 = _vars[i]->getGrad() * _vars[i]->getGrad();
            if (enable_yogi) {
                v[i] = v[i] - grad_pow2 * (1 - second_coef) * (v[i] - grad_pow2).sign();
            } else {
                v[i] = v[i] * second_coef + grad_pow2 * (1 - second_coef);
            }
        }
        for (int i = 0; i < _vars.size(); i++) {
            mm[i] = m[i] / (1 - pow(first_coef, iter_cnt + 1));
            vm[i] = v[i] / (1 - pow(second_coef, iter_cnt + 1));
        }

        // 计算目标更新梯度方向
        for (int i = 0; i < _vars.size(); i++) {
            grads[i] = mm[i] / (vm[i].pow(0.5) + EPSILON);
        }
        if (ENABLE_GRAD_DESCENT_ECHO_GRAD) {
            LOG(INFO) << fmt::format("org grad: {}, adam grad: {}, m: {}, v: {}", gradStr(), gradStr(grads), gradStr(m), gradStr(v));
        }
        if (!checkGrad2rdNorm(grad2rdNorm(grads))) {
            return;
        }
        updateVars(step, grads);
        
        cost = _cost_func(_vars);
        f64 cur_cost = getCost(cost);
        if (!checkCostDiff(pre_cost, cur_cost)) {
            return;
        }
        LOG(INFO) << fmt::format("iter cnt: {}, pre cost: {}, cur cost: {}", iter_cnt, pre_cost, cur_cost);
        pre_cost = cur_cost;
        iter_cnt += 1;
    }

}

}
