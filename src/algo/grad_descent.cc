#include "auto_engine/algo/algo.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cstdlib>
#include <sstream>
#include <vector>

namespace algo {

void OptAlgo::gradDescent() {
    std::srand(std::time(0));

    // 读取超参
    f64 init_step = mustGetHyperParam("init_step");
    f64 step_decay_ratio = mustGetHyperParam("step_decay_ratio");
    f64 tangent_slope_coef = mustGetHyperParam("tangent_slope_coef");
    u32 max_probe_retries = (u32)mustGetHyperParam("max_probe_retries");
    u32 max_iter_cnt = (u32)mustGetHyperParam("max_iter_cnt");

    LOG(INFO) << fmt::format("hyper params, init_step: {}, step_decay_ratio: {}, tangent_slope_coef: {}, max_probe_retries: {}, max_iter_cnt: {}",
        init_step, step_decay_ratio, tangent_slope_coef, max_probe_retries, max_iter_cnt);

    auto find_update_step = [&init_step, &step_decay_ratio, &tangent_slope_coef, &max_probe_retries, this](f64 base, bool& terminate) -> f64 {
        calc_grad();

        f64 res = 0;
        for (int i = 0; i < _vars.size(); i++) {
            auto grad = _vars[i]->getGrad();
            auto grad_pow2 = grad * grad;
            res += grad_pow2.sum().data()[0];
        }

        // 梯度为0，当前已是最低点（或局部最低点）
        if (res < EPSILON) {
            LOG(INFO) << "grad zero, terminate";
            terminate = true;
            return -1;
        }

        // 保存原始tensor，将op中output置空
        std::vector<base::Tensor<f64>> org_outs;
        org_outs.reserve(_vars.size());
        for (int i = 0; i < _vars.size(); i++) {
            org_outs.emplace(org_outs.end(), _vars[i]->getOutput());
        }

        // 随机步长
        auto step = init_step;
        if (ENABLE_GRAD_DESCENNT_RAND_STEP && (std::rand() & ((1 << 10) - 1)) < (1 << 10) * ENABLE_GRAD_DESCENNT_RAND_STEP_RATIO) {
            step = step * ((std::rand() & ((1 << 3) - 1)) + 1);
            LOG(INFO) << "enable grad descent rand step: " << step;
        } 

        // 试探步长，回溯线搜索，找到一个有一定显著值的下降步长
        u32 retries = 0;
        while (retries <= max_probe_retries) {
            f64 target = base - tangent_slope_coef * step * res;
            for (int i = 0; i < _vars.size(); i++) {
                _vars[i]->setOutput(org_outs[i] - _vars[i]->getGrad() * base::Tensor<f64>(_vars[i]->getGrad().shape(), step));
            }
            f64 cost = calc_cost();

            LOG(INFO) << "retries: " << retries << ", base: " << base << ", cost: " << cost;
            if (cost < target) {
                LOG(INFO) << "found step and update: " << step;
                if (ENABLE_GRAD_DESCENT_ECHO_GRAD) {
                    std::stringstream stream;
                    for (int i = 0; i < _vars.size(); i++) {
                        stream << " " << i << "th: " << _vars[i]->getGrad().toString(true);
                    }
                    LOG(INFO) << "grad is:" << stream.str();
                }
                terminate = false;
                return cost;
            }
            step = step * step_decay_ratio;
            retries += 1;
        }
        LOG(INFO) << "step not found, maybe base is opt or curve is too steep";

        // 恢复参数并进行一次传播
        for (int i = 0; i < _vars.size(); i++) {
            _vars[i]->setOutput(org_outs[i]);
        }
        calc_cost();

        terminate = true;
        return -1;
    };
    f64 pre = calc_cost();
    u32 iter_cnt = 0;
    while (iter_cnt < max_iter_cnt) {
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