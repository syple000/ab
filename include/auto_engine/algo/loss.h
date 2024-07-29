#ifndef ALGO_LOSS_H
#define ALGO_LOSS_H

#include "auto_engine/op/op.h"
#include "auto_engine/tensor/tensor.h"
#include <memory>
namespace algo {

class Loss {
public:
    // 第一个参数：模型输出的预测值（一维张量，样本数量）
    // 第二个参数：实际观测到的目标值（常量，一维张量，样本数量）
    // 输出单值张量，标识loss
    static std::shared_ptr<op::Op<base::Tensor<f64>>> mseLoss(std::shared_ptr<op::Op<base::Tensor<f64>>>, const base::Tensor<f64>&); // 最小二乘法

    // 第一个参数：输出的目标概率，下标对应目标（二维张量，样本数量 * 目标数，需要进行一次归一化）
    // 第二个参数：one-hot编码，实际观测到的下标，即目标（一维张量，样本数量，常量）
    // 输出单值张量，标识loss
    static std::shared_ptr<op::Op<base::Tensor<f64>>> crossEntropyLoss(std::shared_ptr<op::Op<base::Tensor<f64>>>, const base::Tensor<f64>&);
};

}

#endif