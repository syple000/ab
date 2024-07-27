#include "auto_engine/algo/grad_descent.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/config/config.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cstdlib>
#include <sstream>
#include <vector>

namespace algo {

void GradDescent::run() {
    // 随机种子
    std::srand(std::time(0));
    
    auto cost_call = [this]() -> f64 { // 返回cost
        calc::Calculator<base::Tensor<f64>> c(_cost_op);
        c.clearOutput();
        c.call();
        const auto& cost = _cost_op->getOutput();
        if (cost.shape().tensorSize() != 1) {
            throw std::runtime_error("cost tensor size != 1");
        }
        return cost.data()[0];
    };
    auto find_update_step = [&cost_call, this](f64 base, bool& terminate) -> f64 {
        // 计算sum(梯度^2) 即梯度的2次范数，和开方
        // 梯度标识曲线收敛陡峭度（dx对dy的影响）
        calc::Calculator<base::Tensor<f64>> c(_cost_op);
        c.clearGrad();
        c.deriv();
        f64 res = 0;
        for (int i = 0; i < _var_vec.size(); i++) {
            auto grad = _var_vec[i]->getGrad();
            auto n = grad.pow(base::Tensor<f64>(grad.shape(), 2))
                .reshape({1, grad.shape().tensorSize()})
                .mmul(base::Tensor<f64>(base::Shape({grad.shape().tensorSize(), 1}), 1))
                .data()[0];
            res += n;
        }
        // 梯度为0，当前已是最低点（或局部最低点）
        if (res < EPSILON) {
            LOG(INFO) << "grad zero, terminate";
            terminate = true;
            return -1;
        }
        // 保存原始tensor，将op中output置空
        std::vector<base::Tensor<f64>> org_outs;
        org_outs.reserve(_var_vec.size());
        for (int i = 0; i < _var_vec.size(); i++) {
            org_outs.emplace(org_outs.end(), _var_vec[i]->getOutput());
        }
        // 试探步长，回溯线搜索，找到一个有一定显著值的下降步长
        auto step = _init_step;
        if (ENABLE_GRAD_DESCENNT_RAND_STEP && (std::rand() & ((1 << 10) - 1)) < (1 << 10) * ENABLE_GRAD_DESCENNT_RAND_STEP_RATIO) {
            step = step * ((std::rand() & ((1 << 3) - 1)) + 1);
            LOG(INFO) << "enable grad descent rand step: " << step;
        } 
        u32 retries = 0;
        while (retries <= _max_probe_retries) {
            f64 target = base - _tangent_slope_coef * step * res;
            for (int i = 0; i < _var_vec.size(); i++) {
                _var_vec[i]->setOutput(org_outs[i] - _var_vec[i]->getGrad() * base::Tensor<f64>(_var_vec[i]->getGrad().shape(), step));
            }
            f64 cost = cost_call();

            LOG(INFO) << "retries: " << retries << ", base: " << base << ", cost: " << cost;
            if (cost < target) {
                LOG(INFO) << "found step and update: " << step;
                if (ENABLE_GRAD_DESCENT_ECHO_GRAD) {
                    std::stringstream stream;
                    for (int i = 0; i < _var_vec.size(); i++) {
                        stream << " " << i << "th: " << _var_vec[i]->getGrad().toString(true);
                    }
                    LOG(INFO) << "grad is: " << stream.str();
                }
                terminate = false;
                return cost;
            }
            step = step * _rho;
            retries += 1;
        }
        LOG(INFO) << "step not found, maybe base is opt or curve is too steep";

        // 恢复参数并进行一次传播
        for (int i = 0; i < _var_vec.size(); i++) {
            _var_vec[i]->setOutput(org_outs[i]);
        }
        cost_call();

        terminate = true;
        return -1;
    };
    f64 pre = cost_call();
    u32 iter_cnt = 0;
    while (iter_cnt < _max_iter_cnt) {
        bool terminate;
        f64 cur = find_update_step(pre, terminate);
        if (terminate) {
            LOG(INFO) << "iter break, terminate...";
            return;
        }
        LOG(INFO) << "iter cnt: " << iter_cnt << ", update, pre: " << pre << ", cur: " << cur;
        pre = cur;
        iter_cnt += 1;
    }
    LOG(INFO) << "iter cnt limit...";
}

}