#include "auto_engine/algo/algo.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <fmt/core.h>
#include <stdexcept>
#include <vector>

namespace algo {

void OptAlgo::adam() {
    // 读取超参
    f64 step = mustGetHyperParam("step");
    f64 first_coef = getHyperParam("first_coef", 0.9);
    f64 second_coef = getHyperParam("second_coef", 0.999);
    u32 max_iter_cnt = round(getHyperParam("max_iter_cnt", 100));

    if (first_coef >= 1 || first_coef <= 0 || second_coef >= 1 || second_coef <= 0) {
        LOG(ERROR) << fmt::format("adam coef invalid, first coef: {}, second coef: {}", first_coef, second_coef);
        throw std::runtime_error(fmt::format("adam coef invalid, first coef: {}, second coef: {}", first_coef, second_coef));
    }

    // 一/二阶矩向量，初始化0
    f64 pre_cost = calcCost();
    std::vector<base::Tensor<f64>> m, v;
    m.reserve(_vars.size()); v.reserve(_vars.size());
    for (int i = 0; i < _vars.size(); i++) {
        m.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
        v.emplace_back(base::Tensor<f64>(_vars[i]->getOutput().shape()));
    }

    // 循环
    u32 iter_cnt = 0;
    while (iter_cnt < max_iter_cnt) {
        calcGrad();

        // 更新并校准矩向量
        for (int i = 0; i < _vars.size(); i++) {
            m[i] = base::Tensor<f64>(m[i].shape(), first_coef) * m[i] + base::Tensor<f64>(m[i].shape(), 1 - first_coef) * _vars[i]->getGrad();
            m[i] = m[i] / base::Tensor<f64>(m[i].shape(), 1 - pow(first_coef, iter_cnt + 1));
            v[i] = base::Tensor<f64>(v[i].shape(), second_coef) * v[i] + base::Tensor<f64>(v[i].shape(), 1 - second_coef) * _vars[i]->getGrad() * _vars[i]->getGrad();
            v[i] = v[i] / base::Tensor<f64>(v[i].shape(), 1 - pow(second_coef, iter_cnt + 1));
        }

        // 计算目标更新梯度方向
        std::vector<base::Tensor<f64>> grads;
        grads.reserve(_vars.size());
        for (int i = 0; i < _vars.size(); i++) {
            grads.emplace_back(m[i] / (v[i].pow(base::Tensor<f64>(v[i].shape(), 0.5)) + base::Tensor<f64>(v[i].shape(), EPSILON)));
        }
        if (ENABLE_GRAD_DESCENT_ECHO_GRAD) {
            LOG(INFO) << fmt::format("org grad: {}, adam grad: {}", gradStr(), gradStr(grads));
        }
        if (!checkGrad2rdNorm(grad2rdNorm(grads))) {
            return;
        }
        updateVars(step, grads);
        
        f64 cur_cost = calcCost();
        if (!checkCostDiff(pre_cost, cur_cost)) {
            return;
        }
        LOG(INFO) << fmt::format("iter cnt: {}, pre cost: {}, cur cost: {}", iter_cnt, pre_cost, cur_cost);
        iter_cnt += 1;
    }

}

}
