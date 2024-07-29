#include "auto_engine/algo/opt.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/config/config.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <cstdlib>
#include <vector>

namespace algo {

void Optimizer::gradDescent() {
    std::srand(std::time(0));

    // 读取超参
    f64 step = mustGetHyperParam("step");
    f64 step_decay_ratio = mustGetHyperParam("step_decay_ratio");
    f64 tangent_slope_coef = mustGetHyperParam("tangent_slope_coef");
    u32 max_probe_retries = round(getHyperParam("max_probe_retries", 100));
    u32 max_iter_cnt = round(getHyperParam("max_iter_cnt", 100));

    LOG(INFO) << fmt::format("hyper params, init_step: {}, step_decay_ratio: {}, tangent_slope_coef: {}, max_probe_retries: {}, max_iter_cnt: {}",
        step, step_decay_ratio, tangent_slope_coef, max_probe_retries, max_iter_cnt);

    auto cost = _cost_func(_vars);
    f64 pre = calcCost(cost);
    auto find_update_step = [&step, &step_decay_ratio, &tangent_slope_coef, &max_probe_retries, &cost, this](f64 base, bool& terminate) -> f64 {
        calcGrad(cost);

        f64 grad_2rd_norm = grad2rdNorm(); 
        if (!checkGrad2rdNorm(grad_2rd_norm)) {terminate = true;return -1;}

        std::vector<base::Tensor<f64>> org_outs = getVars();

        // 随机步长
        auto probe_step = step;
        if (ENABLE_GRAD_DESCENNT_RAND_STEP && (std::rand() & ((1 << 10) - 1)) < (1 << 10) * ENABLE_GRAD_DESCENNT_RAND_STEP_RATIO) {
            probe_step = step * ((std::rand() & ((1 << 3) - 1)) + 1);
            LOG(INFO) << "enable grad descent rand step: " << probe_step;
        } 

        // 试探步长，回溯线搜索，找到一个有一定显著值的下降步长
        u32 retries = 0;
        while (retries <= max_probe_retries) {
            f64 target = base - tangent_slope_coef * probe_step * grad_2rd_norm;
            updateVars(org_outs, probe_step);
            if (!_fix_cost_graph) {
                cost = _cost_func(_vars);
            }
            f64 cost_val = calcCost(cost);

            LOG(INFO) << "retries: " << retries << ", base: " << base << ", cost: " << cost_val;
            if (cost_val < target) {
                LOG(INFO) << "found step and update: " << probe_step;
                if (ENABLE_GRAD_DESCENT_ECHO_GRAD) {
                    LOG(INFO) << "grad is:" << gradStr();
                }
                terminate = false;
                return cost_val;
            }
            probe_step = probe_step * step_decay_ratio;
            retries += 1;
        }
        LOG(INFO) << "step not found, maybe base is opt or curve is too steep";

        restoreVars(std::move(org_outs));
        if (!_fix_cost_graph) {
            cost = _cost_func(_vars);
        }
        calcCost(cost);

        terminate = true;
        return -1;
    };
    u32 iter_cnt = 0;
    while (iter_cnt < max_iter_cnt) {
        bool terminate;
        f64 cur = find_update_step(pre, terminate);
        if (terminate) {
            LOG(INFO) << "iter break, terminate...";
            return;
        }
        if (!checkCostDiff(pre, cur)) {return;}
        LOG(INFO) << "iter cnt: " << iter_cnt << ", update, pre: " << pre << ", cur: " << cur;
        pre = cur;
        iter_cnt += 1;
    }
    LOG(INFO) << "iter cnt limit...";
}

}