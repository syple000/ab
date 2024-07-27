#ifndef ALGO_GRAD_DESCENT_H
#define ALGO_GRAD_DESCENT_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace algo {

class GradDescent {
public:
    // rho：步长更新系数; tangent_slope_coef: 切线斜率系数
    GradDescent(std::shared_ptr<op::Op<base::Tensor<f64>>> cost_op, 
        const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>& var_vec,
        f64 init_step, f64 rho, f64 tangent_slope_coef, u32 max_probe_retries): 
        _cost_op(cost_op), _var_vec(var_vec), _init_step(init_step), _rho(rho), _tangent_slope_coef(tangent_slope_coef), _max_probe_retries(max_probe_retries) {   
        if (_rho <= 0 || _rho >= 1) {
            LOG(ERROR) << "rho should between (0, 1)";
            throw std::runtime_error("rho should between (0, 1)");
        }
        if (_tangent_slope_coef <= 0) {
            LOG(ERROR) << "tangent_slope_coef should gt 0";
            throw std::runtime_error("tangent_slope_coef should gt 0");
        }
    }

    void run();

private:
    std::shared_ptr<op::Op<base::Tensor<f64>>> _cost_op;
    std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>> _var_vec;
    f64 _init_step, _rho, _tangent_slope_coef;
    u32 _max_probe_retries;
};

}

#endif