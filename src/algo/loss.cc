#include "auto_engine/algo/loss.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/sum_expand.h"
#include "auto_engine/shape/shape.h"
#include "auto_engine/tensor/tensor.h"
#include <memory>

namespace algo {

std::shared_ptr<op::Op<base::Tensor<f64>>> Loss::mseLoss(std::shared_ptr<op::Op<base::Tensor<f64>>> outputs, const base::Tensor<f64>& targets) {
    auto diff = std::make_shared<op::Sub<base::Tensor<f64>>>(outputs, std::make_shared<op::DataOp<base::Tensor<f64>>>(targets));
    auto diff_pow2 = std::make_shared<op::Mul<base::Tensor<f64>>>(diff, diff);
    return std::make_shared<op::Sum<base::Tensor<f64>, base::Shape>>(diff_pow2);
}

std::shared_ptr<op::Op<base::Tensor<f64>>> Loss::crossEntropyLoss(std::shared_ptr<op::Op<base::Tensor<f64>>> outputs, const base::Tensor<f64>& targets) {
}

}