#include "auto_engine/algo/grad_descent.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"

namespace algo {

void GradDescent::run() {
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
    auto find_update_step = [&cost_call, this](f64 base) -> f64 {
        // 计算sqrt(sum(梯度^2))
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
        res = ::sqrt(res);
        // 梯度为0，当前已是最低点（或局部最低点）
        if (res < EPSILON) {
            return base;
        }
        // 更新初始步长
        for (int i = 0; i < _var_vec.size(); i++) {
            _var_vec[i]->setOutput(_var_vec[i]->getOutput() - _var_vec[i]->getGrad() * base::Tensor<f64>(_var_vec[i]->getGrad().shape(), _init_step / res));
        }
        // 试探步长
        auto step = _init_step / 2;
        while (step > EPSILON) {
            f64 cost = cost_call();
            LOG(INFO) << "base: " << base << ", cost: " << cost;
            if (cost - base < -_threshold) {
                LOG(INFO) << "found step and update: " << step * 2;
                return cost;
            }
            for (int i = 0; i < _var_vec.size(); i++) {
                _var_vec[i]->setOutput(_var_vec[i]->getOutput() + _var_vec[i]->getGrad() * base::Tensor<f64>(_var_vec[i]->getGrad().shape(), step / res));
            }
            step = step / 2;
        }
        LOG(INFO) << "step not found, maybe base is opt or curve is too steep";
        return base; // 返回base
    };
    f64 pre = cost_call();
    while (true) {
        f64 cur = find_update_step(pre);
        if (abs(cur - pre) < _threshold) {
            LOG(INFO) << "iter break, less than threshold";
            break;
        }
        LOG(INFO) << "update, pre: " << pre << ", cur: " << cur;
        pre = cur;
    }
}

}