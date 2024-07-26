#ifndef ALGO_GRAD_DESCENT_H
#define ALGO_GRAD_DESCENT_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

namespace algo {

class GradDescent {
public:
    GradDescent(std::shared_ptr<op::Op<base::Tensor<f64>>> cost_op, 
        const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>& var_vec,
        f64 threshold, f64 init_step): _cost_op(cost_op), _var_vec(var_vec), _threshold(threshold), _init_step(init_step) {   
    }

    void run();

private:
    std::shared_ptr<op::Op<base::Tensor<f64>>> _cost_op;
    std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>> _var_vec;
    f64 _threshold, _init_step;
};

}

#endif