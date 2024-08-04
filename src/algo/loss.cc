#include "auto_engine/algo/loss.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/add_n.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/div.h"
#include "auto_engine/op/log.h"
#include "auto_engine/op/mmul.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/op/op.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/sum_d_expand_d.h"
#include "auto_engine/op/sum_expand.h"
#include "auto_engine/op/transpose.h"
#include "auto_engine/shape/shape.h"
#include "auto_engine/tensor/tensor.h"
#include <memory>

namespace algo {

std::shared_ptr<op::Op<base::Tensor<f64>>> Loss::mseLoss(std::shared_ptr<op::Op<base::Tensor<f64>>> outputs, std::shared_ptr<op::Op<base::Tensor<f64>>> targets) {
    auto diff = op::Sub<base::Tensor<f64>>::op(outputs, targets);
    auto diff_pow2 = op::Mul<base::Tensor<f64>>::op(diff, diff);
    return op::Sum<base::Tensor<f64>, base::Shape>::op(diff_pow2);
}

std::shared_ptr<op::Op<base::Tensor<f64>>> Loss::crossEntropyLoss(std::shared_ptr<op::Op<base::Tensor<f64>>> outputs, std::shared_ptr<op::Op<base::Tensor<f64>>> targets, u32 classes) {
    // targets.size = outputs.size & outputs[1].size = classes
    
    calc::Calculator<base::Tensor<f64>> c(outputs);
    c.call();
    auto outputs_shape = outputs->getOutput().shape(); // 计算一次。可后续优化

    auto neg_one_hot = targets->getOutput().oneHot(classes).neg(); // targets.size * classes
    auto neg_one_hot_op = op::DataOp<base::Tensor<f64>>::op(neg_one_hot);
    
    auto outputs_pow2 = op::Mul<base::Tensor<f64>>::op(outputs, outputs);
    auto outputs_pow2_sum = op::SumD<base::Tensor<f64>, base::Shape>::op(outputs_pow2, -1); // outputs.size * 1
    auto outputs_pow2_sum_expd = op::ExpandD<base::Tensor<f64>, base::Shape>::op(outputs_pow2_sum, outputs_shape, -1); // 可以增添新的算子来优化，不展开直接除
    auto outputs_pow2_sum_expd_norm = op::AddN<base::Tensor<f64>, base::Shape>::op(outputs_pow2_sum_expd, op::DataOp<base::Tensor<f64>>::op(base::Tensor<f64>(base::Shape({1}), EPSILON)));
    auto norm_outputs = op::Div<base::Tensor<f64>>::op(outputs, outputs_pow2_sum_expd_norm);
    auto trans_norm_outputs = op::Transpose<base::Tensor<f64>>::op(norm_outputs, -1, -2); // classes * outputs.size
    auto log_tnorm_outputs = op::Log<base::Tensor<f64>>::op(trans_norm_outputs);
    auto entropy_mat = op::Mmul<base::Tensor<f64>>::op(trans_norm_outputs, neg_one_hot_op);
    return op::Sum<base::Tensor<f64>, base::Shape>::op(entropy_mat);
}

}