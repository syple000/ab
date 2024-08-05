
#include "auto_engine/algo/opt.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/op/add_n.h"
#include "auto_engine/op/bop.h"
#include "auto_engine/op/cat_split.h"
#include "auto_engine/op/div_n.h"
#include "auto_engine/op/inv.h"
#include "auto_engine/op/mmul.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/div.h"
#include "auto_engine/op/mul_n.h"
#include "auto_engine/op/permute.h"
#include "auto_engine/op/pow_n.h"
#include "auto_engine/op/sub_n.h"
#include "auto_engine/op/sum_d_expand_d.h"
#include "auto_engine/op/sum_expand.h"
#include "auto_engine/op/transpose.h"
#include "auto_engine/shape/shape.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/pow.h"
#include "auto_engine/op/log.h"
#include "auto_engine/op/sin_cos.h"
#include "auto_engine/op/reshape.h"
#include "auto_engine/tensor/tensor.h"
#include "gtest/gtest.h"
#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <glog/logging.h>
#include <ios>
#include <iostream>
#include <memory>
#include <vector>

bool isEq(f64 a, f64 b) {
    if (abs(a-b) < EPSILON) {
        return true;
    }
    LOG(ERROR) << "a: " << a << ", b: " << b << " diff";
    return false;
}

bool isEq(const base::Tensor<f64>& a, const base::Tensor<f64>& b) {
    return a == b;
}

TEST(Test_grad_f64, Test){
    std::cout.precision(8);
    std::cout << std::fixed;
    // 测试函数f(x1, x2) = (2 * x1 + x2) * (x1 - x2) + x2/x1 + x2^(x2 + x1 + 1) + log(x2 + x1) + sin(x2-x1)/cos(x2+x1)
    auto x1 = op::DataOp<f64>::op(3, true);
    auto x2 = op::DataOp<f64>::op(4, true);
    auto c1 = op::DataOp<f64>::op(1);
    auto c2 = op::DataOp<f64>::op(2);
    auto item1 = op::Mul<f64>::op(c2, x1);
    auto item2 = op::Add<f64>::op(item1, x2);
    auto item3 = op::Sub<f64>::op(x1, x2);
    auto item4 = op::Mul<f64>::op(item2, item3);
    auto item5 = op::Div<f64>::op(x2, x1);
    auto item6 = op::Add<f64>::op(item4, item5);
    auto item7 = op::Add<f64>::op(x2, x1);
    auto item8 = op::Add<f64>::op(item7, c1);
    auto item9 = op::Pow<f64>::op(x2, item8);
    auto item10 = op::Add<f64>::op(item6, item9);
    auto item11 = op::Log<f64>::op(item7);
    auto item12 = op::Add<f64>::op(item10, item11);

    auto item13 = op::Sub<f64>::op(x2, x1);
    auto item14 = op::Sin<f64>::op(item13);
    auto item15 = op::Cos<f64>::op(item7);
    auto item16 = op::Div<f64>::op(item14, item15);

    auto item = op::Add<f64>::op(item12, item16);

    ASSERT_TRUE(isEq(item->getOutput(), 65530.39539744));
    auto c = calc::Calculator<f64>(item);
    c.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 90860.14165895));
    ASSERT_TRUE(isEq(x2->getGrad(), 221915.35278516));
    c.clearGrad();
    c.createGradGraph();
    auto x1_grad  = x1->getGradGraph();
    auto x2_grad = x2->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(isEq(x1_grad->getOutput(), 90860.14165895));
    ASSERT_TRUE(isEq(x2_grad->getOutput(), 221915.35278516));
    auto cx1 = calc::Calculator<f64>(x1_grad);
    auto cx2 = calc::Calculator<f64>(x2_grad);
    cx1.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 125952.59694293));
    ASSERT_TRUE(isEq(x2->getGrad(), 324039.04543274));
    cx1.clearGrad();
    cx2.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 324039.04543274));
    ASSERT_TRUE(isEq(x2->getGrad(), 751501.54782524));
    cx2.clearGrad();
}

TEST(Test_grad_tensor, Test){
    std::cout.precision(8);
    std::cout << std::fixed;
    // 测试函数f(x1, x2) = (2 * x1 + x2) * (x1 - x2) + x2/x1 + x2^(x2 + x1 + 1) + log(x2 + x1) + sin(x2-x1)/cos(x2+x1)
    auto t1 = base::Tensor<f64>(base::Shape({2}), {3, 1});
    auto t2 = base::Tensor<f64>(base::Shape({2}), {4, 2});
    auto ct1 = base::Tensor<f64>(base::Shape({2}), 1);
    auto ct2 = base::Tensor<f64>(base::Shape({2}), 2);
    auto x1 = op::DataOp<base::Tensor<f64>>::op(t1, true);
    auto x2 = op::DataOp<base::Tensor<f64>>::op(t2, true);
    auto c1 = op::DataOp<base::Tensor<f64>>::op(ct1);
    auto c2 = op::DataOp<base::Tensor<f64>>::op(ct2);
    auto item1 = op::Mul<base::Tensor<f64>>::op(c2, x1);
    auto item2 = op::Add<base::Tensor<f64>>::op(item1, x2);
    auto item3 = op::Sub<base::Tensor<f64>>::op(x1, x2);
    auto item4 = op::Mul<base::Tensor<f64>>::op(item2, item3);
    auto item5 = op::Div<base::Tensor<f64>>::op(x2, x1);
    auto item6 = op::Add<base::Tensor<f64>>::op(item4, item5);
    auto item7 = op::Add<base::Tensor<f64>>::op(x2, x1);
    auto item8 = op::Add<base::Tensor<f64>>::op(item7, c1);
    auto item9 = op::Pow<base::Tensor<f64>>::op(x2, item8);
    auto item10 = op::Add<base::Tensor<f64>>::op(item6, item9);
    auto item11 = op::Log<base::Tensor<f64>>::op(item7);
    auto item12 = op::Add<base::Tensor<f64>>::op(item10, item11);

    auto item13 = op::Sub<base::Tensor<f64>>::op(x2, x1);
    auto item14 = op::Sin<base::Tensor<f64>>::op(item13);
    auto item15 = op::Cos<base::Tensor<f64>>::op(item7);
    auto item16 = op::Div<base::Tensor<f64>>::op(item14, item15);

    auto item = op::Add<base::Tensor<f64>>::op(item12, item16);

    ASSERT_TRUE(isEq(item->getOutput(), base::Tensor<f64>(base::Shape({2}), {65530.39539744, 14.24863515})));
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    c.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), base::Tensor<f64>(base::Shape({2}), {90860.14165895, 12.09061357})));
    ASSERT_TRUE(isEq(x2->getGrad(), base::Tensor<f64>(base::Shape({2}), {221915.35278516, 38.99908548})));
    c.clearGrad();
    c.createGradGraph();
    auto x1_grad = x1->getGradGraph();
    auto x2_grad = x2->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(isEq(x1_grad->getOutput(), base::Tensor<f64>(base::Shape({2}), {90860.14165895, 12.09061357})));
    ASSERT_TRUE(isEq(x2_grad->getOutput(), base::Tensor<f64>(base::Shape({2}), {221915.35278516, 38.99908548})));
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1_grad);
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2_grad);
    cx1.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), base::Tensor<f64>(base::Shape({2}), {125952.59694293, 15.38600131})));
    ASSERT_TRUE(isEq(x2->getGrad(), base::Tensor<f64>(base::Shape({2}), {324039.04543274, 34.02235037})));
    cx1.clearGrad();
    cx2.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), base::Tensor<f64>(base::Shape({2}), {324039.04543274, 34.02235037})));
    ASSERT_TRUE(isEq(x2->getGrad(), base::Tensor<f64>(base::Shape({2}), {751501.54782524, 114.05860797})));
    cx2.clearGrad();
}

TEST(Test_grad_tensor3, test) {
    auto t1 = base::Tensor<f64>(base::Shape({2, 2}), {1, 2, 3, 4});
    auto t2 = base::Tensor<f64>(base::Shape({1, 1}), {2});
    auto x1 = op::DataOp<base::Tensor<f64>>::op(t1, true);
    auto x2 = op::DataOp<base::Tensor<f64>>::op(t2, true);
    auto item1 = op::AddN<base::Tensor<f64>, base::Shape>::op(x1, x2);
    auto item2 = op::MulN<base::Tensor<f64>, base::Shape>::op(item1, x2);
    auto item3 = op::PowN<base::Tensor<f64>, base::Shape>::op(item2, x2);
    auto item4 = op::DivN<base::Tensor<f64>, base::Shape>::op(item3, x2);
    auto item = op::SubN<base::Tensor<f64>, base::Shape>::op(item4, x2);
    ASSERT_TRUE(item->getOutput() == base::Tensor<f64>(base::Shape({2, 2}), {16, 30, 48, 70}));
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    c.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 2}), {12, 16, 20, 24}));
    ASSERT_TRUE(x2->getGrad() == base::Tensor<f64>(base::Shape({1, 1}), {546.836333214}));
    c.clearGrad();
    c.createGradGraph();
    auto x1_grad = x1->getGradGraph();
    auto x2_grad = x2->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(x1_grad->getOutput() == base::Tensor<f64>(base::Shape({2, 2}), {12, 16, 20, 24}));
    ASSERT_TRUE(x2_grad->getOutput() == base::Tensor<f64>(base::Shape({1, 1}), {546.836333214}));
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1_grad);
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2_grad);
    cx1.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 2}), {4, 4, 4, 4}));
    ASSERT_TRUE(x2->getGrad() == base::Tensor<f64>(base::Shape({1, 1}), {248.461639752}));
    cx1.clearGrad();
    cx2.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 2}), {37.5011136307, 53.2710646669, 70.0517018599, 87.6377595949}));
    ASSERT_TRUE(x2->getGrad() == base::Tensor<f64>(base::Shape({1, 1}), {1951.59501836}));
    cx2.clearGrad();
}

TEST(Test_grad_test4, test) {
    auto t1 = base::Tensor<f64>(base::Shape({2, 3, 3}), {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.5});
    auto x1 = op::DataOp<base::Tensor<f64>>::op(t1, true);
    auto item1_1 = op::Permute<base::Tensor<f64>>::op(x1, std::vector<u32>{2, 0, 1});
    auto item1_2 = op::Permute<base::Tensor<f64>>::op(item1_1, std::vector<u32>{1, 0, 2});
    auto item1 = op::SumD<base::Tensor<f64>, base::Shape>::op(item1_2, 0);
    auto item2 = op::SumD<base::Tensor<f64>, base::Shape>::op(x1, 1);
    auto item3 = op::SumD<base::Tensor<f64>, base::Shape>::op(x1, -1);
    auto item4 = op::Mmul<base::Tensor<f64>>::op(op::Transpose<base::Tensor<f64>>::op(item2, 0, 1), item3);
    auto item5_ = op::Mmul<base::Tensor<f64>>::op(item1, item4);
    auto item5 = op::Permute<base::Tensor<f64>>::op(item5_, std::vector<u32>{1, 0});
    auto item = op::Sum<base::Tensor<f64>, base::Shape>::op(op::Mul<base::Tensor<f64>>::op(item5, item5));
    ASSERT_TRUE(item->getOutput() == base::Tensor<f64>(base::Shape({1}), {1062.682578000}));
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    c.deriv();
    auto grad = x1->getGrad();
    ASSERT_TRUE(grad == base::Tensor<f64>(base::Shape({2, 3, 3}), {
        374.643360000, 597.470040000, 782.094420000,
        576.224280000, 804.431520000, 994.436460000,
        754.919460000, 988.507260000, 1183.892760000,
        323.155440000, 498.101400000, 644.233500000,
        475.332840000, 655.659360000, 807.172020000,
        611.420940000, 797.128020000, 954.021240000,
    }));
    c.clearGrad();
    c.createGradGraph();
    auto xgrad = x1->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(xgrad->getOutput() == grad);
    auto cx1 = calc::Calculator<base::Tensor<f64>>(xgrad);
    cx1.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 3}), {
        4685.720400000, 6424.606800000, 7880.857200000,
        6343.822800000, 8114.763600000, 9603.068400000,
        7801.088400000, 9604.083600000, 11124.442800000,
        4047.548400000, 5488.614000000, 6702.901200000,
        5399.017200000, 6872.137200000, 8118.478800000,
        6595.311600000, 8100.486000000, 9378.882000000,
    }));
    cx1.clearGrad();
}

TEST(Test_grad_tensor5, test) {
    auto t1 = base::Tensor<f64>(base::Shape({5, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    auto x1 = op::DataOp<base::Tensor<f64>>::op(t1, true);
    auto v0_3 = op::Split<base::Tensor<f64>>::op(x1, {2, 2, 1}, 0);
    auto v2_2 = op::Split<base::Tensor<f64>>::op(x1, {1, 1}, 2);
    auto item1 = op::Add<base::Tensor<f64>>::op(v0_3[0], v0_3[1]);
    auto item2 = op::Transpose<base::Tensor<f64>>::op(item1, -1, 0);
    auto item3 = op::Cat<base::Tensor<f64>>::op({item2, v0_3[2]}, 0); // 3 * 2 * 2
    auto item4 = op::Mul<base::Tensor<f64>>::op(v2_2[0], v2_2[1]);
    auto item5 = op::Div<base::Tensor<f64>>::op(v2_2[0], v2_2[1]);
    auto item6 = op::Cat<base::Tensor<f64>>::op({item4, item5}, 2); // 5 * 2 * 2
    auto v0_3_1 = op::Split<base::Tensor<f64>>::op(item6, {3, 2}, 0)[0];
    auto item7 = op::Mmul<base::Tensor<f64>>::op(v0_3_1, item3);
    auto item8 = op::Inv<base::Tensor<f64>>::op(item7);
    auto item = op::Sum<base::Tensor<f64>, base::Shape>::op(item8);
    calc::Calculator<base::Tensor<f64>> c(item);
    c.deriv();
    auto grad = x1->getGrad();
    ASSERT_TRUE(grad == base::Tensor<f64>(base::Shape({5, 2, 2}), {
        -0.386284722, 0.632232615,
        -0.275752315, -0.280380763,
        -0.318973214, 0.010641399,
        0.233623360, 0.061661808,
        0.118625491, 0.397573698,
        -0.208224419, -0.316348220,
        -0.179687500, -0.119642857,
        0.179687500, 0.119642857,
        5.208907254, -4.919536272,
        -5.208907254, 4.919536272,
    }));
    c.clearGrad();
    c.createGradGraph();
    auto xgrad = x1->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(grad == xgrad->getOutput());
    calc::Calculator<base::Tensor<f64>> cx1(xgrad);
    cx1.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({5, 2, 2}), {
        0.823495370, -0.374829358,
        0.144097222, 0.027607136,
        0.082348356, -0.029297298,
        -0.063116786, 0.015117789,
        -0.132096604, 0.003271449,
        0.142764239, -0.012139030,
        0.064236111, -0.015006378,
        -0.064236111, 0.015006378,
        0.213942686, -0.218129411,
        -0.213942687, 0.218129412,
    }));
    cx1.clearGrad();
}

TEST(Test_grad_tensor2, test) {
    // 测试：m1 = [[1, 2], [3, 4]], m2 = [[2, 3, 4], [5, 6, 7]]
    // f(m1, m2) = Transpose(inv(Transpose(m1*m2) * m2) * Transpose([1,1,1])) * Transpose([1, 1, 1])
    auto t1 = base::Tensor<f64>(base::Shape({2, 2, 2}), {1, 2, 3, 4, 1, 2, 2, 1});
    auto t2 = base::Tensor<f64>(base::Shape({2, 3, 2}), {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1});
    auto t3 = base::Tensor<f64>(base::Shape({2, 1, 3}), {1, 1, 1, 1, 1, 1});
    auto x1_ = op::DataOp<base::Tensor<f64>>::op(t1, true);
    auto x1 = op::Inv<base::Tensor<f64>>::op(x1_);
    auto x2_ = op::DataOp<base::Tensor<f64>>::op(t2, true);
    auto x2 = op::Reshape<base::Tensor<f64>, base::Shape>::op(x2_, base::Shape({2, 2, 3}));
    auto ct = op::DataOp<base::Tensor<f64>>::op(t3);
    auto item1 = op::Mmul<base::Tensor<f64>>::op(x1, x2);
    auto item2 = op::Transpose<base::Tensor<f64>>::op(item1, -2, -1);
    auto item3 = op::Mmul<base::Tensor<f64>>::op(item2, x2);
    auto item = op::Sum<base::Tensor<f64>, base::Shape>::op(item3);
    ASSERT_TRUE(item->getOutput() == base::Tensor<f64>(base::Shape({1}), {73.5}));
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    c.deriv();
    ASSERT_TRUE(x1_->getGrad() == base::Tensor<f64>(base::Shape({2, 2, 2}), {-31.5, -15.75, 4.5, 2.25, -1, 8, 8, -64}));
    ASSERT_TRUE(x2_->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 2}), {13.5, 13.5, 13.5, 0, 0, 0, -2, -2, -2, 16, 16, 16}));
    c.clearGrad();
    c.createGradGraph();
    auto x1_grad = x1_->getGradGraph();
    auto x2_grad = x2_->getGradGraph();
    c.clearGradGraph();
    ASSERT_TRUE(x1_grad->getOutput() == base::Tensor<f64>(base::Shape({2, 2, 2}), {-31.5, -15.75, 4.5, 2.25, -1, 8, 8, -64}));
    ASSERT_TRUE(x2_grad->getOutput() == base::Tensor<f64>(base::Shape({2, 3, 2}), {13.5, 13.5, 13.5, 0, 0, 0, -2, -2, -2, 16, 16, 16}));
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1_grad);
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2_grad);
    cx1.deriv();
    ASSERT_TRUE(x1_->getGrad() == base::Tensor<f64>(base::Shape({2, 2, 2}), {-60.75, 40.5, 20.25, 0, -4-2.0/3, 16+1.0/3, 16+1.0/3, 37+1.0/3}));
    ASSERT_TRUE(x2_->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 2}), {9, 9, 9, -9, -9, -9, -4-2.0/3, -4-2.0/3, -4-2.0/3, -4-2.0/3, -4-2.0/3, -4-2.0/3}));
    cx1.clearGrad();
    cx2.deriv();
    ASSERT_TRUE(x1_->getGrad() == base::Tensor<f64>(base::Shape({2, 2, 2}), {36, -29.25, -9, 2.25, 2, -7, -7, -16}));
    ASSERT_TRUE(x2_->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 2}), {-4.5, -4.5, -4.5, 4.5, 4.5, 4.5, 2, 2, 2, 2, 2, 2}));
    cx2.clearGrad();
}

TEST(Test_tensor, test) {
    base::Tensor<f64> t1(base::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(t1.transpose(-2, -1) == base::Tensor<f64>(base::Shape({3, 2}), {1, 4, 2, 5, 3, 6}));
    base::Tensor<f64> t2(base::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(t2.transpose(-2, -1) == base::Tensor<f64>(base::Shape({2, 3, 2}), {1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6}));
    ASSERT_TRUE(t2.transpose(0, 1) == base::Tensor<f64>(base::Shape({2, 2, 3}), {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}));
    base::Tensor<f64> t3(base::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(t1.mmul(t3) == base::Tensor<f64>(base::Shape({2, 2}), {22, 28, 49, 64}));
    base::Tensor<f64> t4(base::Shape({2, 3, 2}), {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(t2.mmul(t4) == base::Tensor<f64>(base::Shape({2, 2, 2}), {22, 28, 49, 64, 22, 28, 49, 64}));
    base::Tensor<f64> t5(base::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(t4.mmul(t5) == base::Tensor<f64>(base::Shape({2, 3, 3}), {9, 12, 15, 19, 26, 33, 29, 40, 51, 9, 12, 15, 19, 26, 33, 29, 40, 51}));
    base::Tensor<f64> t6(base::Shape({3, 3}), {1, 2, 3, 4, 5, 6, 7, 2, 9});
    ASSERT_TRUE(t6.inv() == base::Tensor<f64>(base::Shape({3, 3}), {-11.0/12, 1.0/3, 1.0/12, -1.0/6, 1.0/3, -1.0/6, 3.0/4, -1.0/3, 1.0/12}));
    base::Tensor<f64> t7(base::Shape({2, 3, 3}), {1, 2, 3, 4, 5, 6, 7, 2, 9, 0, 2, 3, 4, 5, 6, 7, 2, 9});
    ASSERT_TRUE(t7.inv() == base::Tensor<f64>(base::Shape({2, 3, 3}), {-11.0/12, 1.0/3, 1.0/12, -1.0/6, 1.0/3, -1.0/6, 3.0/4, -1.0/3, 1.0/12, -11.0/23, 4.0/23, 1.0/23, -2.0/23, 7.0/23, -4.0/23, 9.0/23, -14.0/69, 8.0/69}));
    ASSERT_TRUE(t2.sum() == base::Tensor<f64>(base::Shape({1}), 42));
    ASSERT_TRUE(t2.sum().expand(t2.shape()) == base::Tensor<f64>(base::Shape({2, 2, 3}), {42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
    ASSERT_TRUE(t2.sum(0) == base::Tensor<f64>(base::Shape({2, 3}), {2, 4, 6, 8, 10, 12}));
    ASSERT_TRUE(t2.sum(-1) == base::Tensor<f64>(base::Shape({2, 2}), {6, 15, 6, 15}));
    ASSERT_TRUE(t2.sum(0).expand(t2.shape(), 0) == base::Tensor<f64>(base::Shape({2, 2, 3}), {2, 4, 6, 8, 10, 12, 2, 4, 6, 8, 10, 12}));
    ASSERT_TRUE(t2.sum(-1).expand(t2.shape(), -1) == base::Tensor<f64>(base::Shape({2, 2, 3}), {6, 6, 6, 15, 15, 15, 6, 6, 6, 15, 15, 15}));
    ASSERT_TRUE(t1 + 1 == base::Tensor<f64>(base::Shape({2, 3}), {2, 3, 4, 5, 6, 7}));
    ASSERT_TRUE(t1 - 1 == base::Tensor<f64>(base::Shape({2, 3}), {0, 1, 2, 3, 4, 5}));
    ASSERT_TRUE(t1 * 2 == base::Tensor<f64>(base::Shape({2, 3}), {2, 4, 6, 8, 10, 12}));
    ASSERT_TRUE(t1 / 2 == base::Tensor<f64>(base::Shape({2, 3}), {0.5, 1, 1.5, 2, 2.5, 3}));
    ASSERT_TRUE(t1.pow(2) == base::Tensor<f64>(base::Shape({2, 3}), {1, 4, 9, 16, 25, 36}));

    base::Tensor<f64> t = base::Tensor<f64>(base::Shape({5}), {0, 1, 3, 2, 1});
    ASSERT_TRUE(t.oneHot(4) == base::Tensor<f64>(base::Shape({5, 4}), {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 0, 0
    }));
    ASSERT_TRUE(t2.permute({2, 0, 1}) == base::Tensor<f64>(base::Shape({3, 2, 2}), {1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6}));
    ASSERT_TRUE(t2.permute({2, 1, 0}) == base::Tensor<f64>(base::Shape({3, 2, 2}), {1, 1, 4, 4, 2, 2, 5, 5, 3, 3, 6, 6}));
    ASSERT_TRUE(t2.permute({1, 0, 2}) == base::Tensor<f64>(base::Shape({2, 2, 3}), {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}));
    base::Tensor<f64> t10 = base::Tensor<f64>(base::Shape({2, 1, 2}), {1, 2, 3, 4});
    base::Tensor<f64> t11 = base::Tensor<f64>(base::Shape({2, 1, 2}), {5, 6, 7, 8});
    base::Tensor<f64> t12 = base::Tensor<f64>(base::Shape({1, 1, 2}), {9, 10});
    base::Tensor<f64> t13 = base::Tensor<f64>(base::Shape({2, 2, 2}), {11, 12, 13, 14, 15, 16, 17, 18});
    auto t14 = base::Tensor<f64>::cat({t10, t11, t12}, 0);
    auto t15 = base::Tensor<f64>::cat({t10, t11, t13}, 1);
    ASSERT_TRUE(t14 == base::Tensor<f64>(base::Tensor<f64>(base::Shape({5, 1, 2}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})));
    ASSERT_TRUE(t15 == base::Tensor<f64>(base::Shape({2, 4, 2}), {1, 2, 5, 6, 11, 12, 13, 14, 3, 4, 7, 8, 15, 16, 17, 18}));

    auto v1 = base::Tensor<f64>::split(t14, {2, 2, 1}, 0);
    ASSERT_TRUE(t14.split(base::Shape({2, 1, 2}), 0, 0) == v1[0]);
    ASSERT_TRUE(t14.split(base::Shape({2, 1, 2}), 0, 2) == v1[1]);
    ASSERT_TRUE(t14.split(base::Shape({1, 1, 2}), 0, 4) == v1[2]);
    auto v2 = base::Tensor<f64>::split(t15, {1, 1, 2}, 1);
    ASSERT_TRUE(t15.split(base::Shape({2, 1, 2}), 1, 0) == v2[0]);
    ASSERT_TRUE(t15.split(base::Shape({2, 1, 2}), 1, 1) == v2[1]);
    ASSERT_TRUE(t15.split(base::Shape({2, 2, 2}), 1, 2) == v2[2]);

    auto t16 = v1[1].cat(t14.shape(), 0, 2);
    t16 = t16 + v1[0].cat(t14.shape(), 0, 0);
    t16 = t16 + v1[2].cat(t14.shape(), 0, 4);
    auto t17 = v2[1].cat(t15.shape(), 1, 1);
    t17 = t17 + v2[0].cat(t15.shape(), 1, 0);
    t17 = t17 + v2[2].cat(t15.shape(), 1, 2);
    ASSERT_TRUE(t16 == base::Tensor<f64>(base::Tensor<f64>(base::Shape({5, 1, 2}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})));
    ASSERT_TRUE(t17 == base::Tensor<f64>(base::Shape({2, 4, 2}), {1, 2, 5, 6, 11, 12, 13, 14, 3, 4, 7, 8, 15, 16, 17, 18}));
}

TEST(Test_grad_descent, test) {
    auto t1 = base::Tensor<f64>(base::Shape({2}), {1, 2});
    auto x = op::DataOp<base::Tensor<f64>>::op(t1, true);

    auto cost_func = [](const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>& args) {
        auto t2 = base::Tensor<f64>(base::Shape({2}), {2, 3});
        auto t3 = base::Tensor<f64>(base::Shape({2}), {4, 5});
        auto ct1 = op::DataOp<base::Tensor<f64>>::op(t2);
        auto ct2 = op::DataOp<base::Tensor<f64>>::op(t3);
        auto item1 = op::Mul<base::Tensor<f64>>::op(args[0], args[0]);
        auto item2 = op::Mul<base::Tensor<f64>>::op(ct1, args[0]);
        auto item3 = op::Add<base::Tensor<f64>>::op(item1, item2);
        auto item4 = op::Add<base::Tensor<f64>>::op(item3, ct2);
        auto item5 = op::Reshape<base::Tensor<f64>, base::Shape>::op(item4, base::Shape({1, 2}));
        auto item6 = op::DataOp<base::Tensor<f64>>::op(base::Tensor<f64>(base::Shape({2, 1}), 1));
        auto item = op::Mmul<base::Tensor<f64>>::op(item5, item6);
        return item; 
    };
   
    auto d1 = std::make_shared<algo::Optimizer>(cost_func, std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>{x});
    d1->algoHyperParams("grad_descent", {
        {"step", 1},
        {"step_decay_ratio", 0.8},
        {"tangent_slope_coef", 0.0001},
        {"max_probe_retries", 50},
        {"max_iter_cnt", 50}
    });
    d1->run();
    std::cout << x->getOutput().toString() << std::endl;

    auto d2 = std::make_shared<algo::Optimizer>(cost_func, std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>{x});
    d2->algoHyperParams("adam", {
        {"step", 0.001},
        // {"enable_yogi", 1}
    });
    d2->run();
    std::cout << x->getOutput().toString() << std::endl;
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "./info.log");
    google::SetLogDestination(google::ERROR, "./error.log");
    google::SetLogDestination(google::WARNING, "./warning.log");

    cuda::init();

    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();

    google::ShutdownGoogleLogging();
    return 0;
}