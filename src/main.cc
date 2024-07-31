
#include "auto_engine/algo/opt.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/op/add_n.h"
#include "auto_engine/op/bop.h"
#include "auto_engine/op/add.h"
#include "auto_engine/op/div_n.h"
#include "auto_engine/op/inv.h"
#include "auto_engine/op/mmul.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/div.h"
#include "auto_engine/op/mul_n.h"
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
    auto x1 = std::make_shared<op::DataOp<f64>>(3, true);
    auto x2 = std::make_shared<op::DataOp<f64>>(4, true);
    auto c1 = std::make_shared<op::DataOp<f64>>(1);
    auto c2 = std::make_shared<op::DataOp<f64>>(2);
    auto item1 = std::make_shared<op::Mul<f64>>(c2, x1);
    auto item2 = std::make_shared<op::Add<f64>>(item1, x2);
    auto item3 = std::make_shared<op::Sub<f64>>(x1, x2);
    auto item4 = std::make_shared<op::Mul<f64>>(item2, item3);
    auto item5 = std::make_shared<op::Div<f64>>(x2, x1);
    auto item6 = std::make_shared<op::Add<f64>>(item4, item5);
    auto item7 = std::make_shared<op::Add<f64>>(x2, x1);
    auto item8 = std::make_shared<op::Add<f64>>(item7, c1);
    auto item9 = std::make_shared<op::Pow<f64>>(x2, item8);
    auto item10 = std::make_shared<op::Add<f64>>(item6, item9);
    auto item11 = std::make_shared<op::Log<f64>>(item7);
    auto item12 = std::make_shared<op::Add<f64>>(item10, item11);

    auto item13 = std::make_shared<op::Sub<f64>>(x2, x1);
    auto item14 = std::make_shared<op::Sin<f64>>(item13);
    auto item15 = std::make_shared<op::Cos<f64>>(item7);
    auto item16 = std::make_shared<op::Div<f64>>(item14, item15);

    auto item = std::make_shared<op::Add<f64>>(item12, item16);

    auto c = calc::Calculator<f64>(item);
    ASSERT_TRUE(isEq(c.call(), 65530.39539744));
    // std::cout << c.call() << std::endl;
    c.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 90860.14165895));
    // std::cout << x1->getGrad() << std::endl;
    ASSERT_TRUE(isEq(x2->getGrad(), 221915.35278516));
    // std::cout << x2->getGrad() << std::endl;
    c.clearGrad();
    c.createGradGraph();
    auto cx1 = calc::Calculator<f64>(x1->getGradGraph());
    auto cx2 = calc::Calculator<f64>(x2->getGradGraph());
    c.clearGradGraph();
    ASSERT_TRUE(isEq(cx1.call(), 90860.14165895));
    // std::cout << cx1.call() << std::endl;
    ASSERT_TRUE(isEq(cx2.call(), 221915.35278516));
    // std::cout << cx2.call() << std::endl;
    cx1.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 125952.59694293));
    // std::cout << x1->getGrad() << std::endl;
    ASSERT_TRUE(isEq(x2->getGrad(), 324039.04543274));
    // std::cout << x2->getGrad() << std::endl;
    cx1.clearGrad();
    cx2.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), 324039.04543274));
    // std::cout << x1->getGrad() << std::endl;
    ASSERT_TRUE(isEq(x2->getGrad(), 751501.54782524));
    // std::cout << x2->getGrad() << std::endl;
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
    auto x1 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t1, true);
    auto x2 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t2, true);
    auto c1 = std::make_shared<op::DataOp<base::Tensor<f64>>>(ct1);
    auto c2 = std::make_shared<op::DataOp<base::Tensor<f64>>>(ct2);
    auto item1 = std::make_shared<op::Mul<base::Tensor<f64>>>(c2, x1);
    auto item2 = std::make_shared<op::Add<base::Tensor<f64>>>(item1, x2);
    auto item3 = std::make_shared<op::Sub<base::Tensor<f64>>>(x1, x2);
    auto item4 = std::make_shared<op::Mul<base::Tensor<f64>>>(item2, item3);
    auto item5 = std::make_shared<op::Div<base::Tensor<f64>>>(x2, x1);
    auto item6 = std::make_shared<op::Add<base::Tensor<f64>>>(item4, item5);
    auto item7 = std::make_shared<op::Add<base::Tensor<f64>>>(x2, x1);
    auto item8 = std::make_shared<op::Add<base::Tensor<f64>>>(item7, c1);
    auto item9 = std::make_shared<op::Pow<base::Tensor<f64>>>(x2, item8);
    auto item10 = std::make_shared<op::Add<base::Tensor<f64>>>(item6, item9);
    auto item11 = std::make_shared<op::Log<base::Tensor<f64>>>(item7);
    auto item12 = std::make_shared<op::Add<base::Tensor<f64>>>(item10, item11);

    auto item13 = std::make_shared<op::Sub<base::Tensor<f64>>>(x2, x1);
    auto item14 = std::make_shared<op::Sin<base::Tensor<f64>>>(item13);
    auto item15 = std::make_shared<op::Cos<base::Tensor<f64>>>(item7);
    auto item16 = std::make_shared<op::Div<base::Tensor<f64>>>(item14, item15);

    auto item = std::make_shared<op::Add<base::Tensor<f64>>>(item12, item16);

    auto c = calc::Calculator<base::Tensor<f64>>(item);
    ASSERT_TRUE(isEq(c.call(), base::Tensor<f64>(base::Shape({2}), {65530.39539744, 14.24863515})));
    c.deriv();
    ASSERT_TRUE(isEq(x1->getGrad(), base::Tensor<f64>(base::Shape({2}), {90860.14165895, 12.09061357})));
    ASSERT_TRUE(isEq(x2->getGrad(), base::Tensor<f64>(base::Shape({2}), {221915.35278516, 38.99908548})));
    c.clearGrad();
    c.createGradGraph();
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1->getGradGraph());
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2->getGradGraph());
    c.clearGradGraph();
    ASSERT_TRUE(isEq(cx1.call(), base::Tensor<f64>(base::Shape({2}), {90860.14165895, 12.09061357})));
    ASSERT_TRUE(isEq(cx2.call(), base::Tensor<f64>(base::Shape({2}), {221915.35278516, 38.99908548})));
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
    auto x1 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t1, true);
    auto x2 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t2, true);
    auto item1 = std::make_shared<op::AddN<base::Tensor<f64>, base::Shape>>(x1, x2);
    auto item2 = std::make_shared<op::MulN<base::Tensor<f64>, base::Shape>>(item1, x2);
    auto item3 = std::make_shared<op::PowN<base::Tensor<f64>, base::Shape>>(item2, x2);
    auto item4 = std::make_shared<op::DivN<base::Tensor<f64>, base::Shape>>(item3, x2);
    auto item = std::make_shared<op::SubN<base::Tensor<f64>, base::Shape>>(item4, x2);
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    ASSERT_TRUE(c.call() == base::Tensor<f64>(base::Shape({2, 2}), {16, 30, 48, 70}));
    c.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 2}), {12, 16, 20, 24}));
    ASSERT_TRUE(x2->getGrad() == base::Tensor<f64>(base::Shape({1, 1}), {546.836333214}));
    c.clearGrad();
    c.createGradGraph();
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1->getGradGraph());
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2->getGradGraph());
    c.clearGradGraph();
    ASSERT_TRUE(cx1.call() == base::Tensor<f64>(base::Shape({2, 2}), {12, 16, 20, 24}));
    ASSERT_TRUE(cx2.call() == base::Tensor<f64>(base::Shape({1, 1}), {546.836333214}));
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
    auto x1 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t1, true);
    auto item1 = std::make_shared<op::SumD<base::Tensor<f64>, base::Shape>>(x1, 0);
    auto item2 = std::make_shared<op::SumD<base::Tensor<f64>, base::Shape>>(x1, 1);
    auto item3 = std::make_shared<op::SumD<base::Tensor<f64>, base::Shape>>(x1, -1);
    auto item4 = std::make_shared<op::Mmul<base::Tensor<f64>>>(std::make_shared<op::Transpose<base::Tensor<f64>>>(item2, 0, 1), item3);
    auto item5 = std::make_shared<op::Mmul<base::Tensor<f64>>>(item1, item4);
    auto item = std::make_shared<op::Sum<base::Tensor<f64>, base::Shape>>(std::make_shared<op::Mul<base::Tensor<f64>>>(item5, item5));
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    ASSERT_TRUE(c.call() == base::Tensor<f64>(base::Shape({1}), {1189.891764000}));
    c.deriv();
    auto xgrad = x1->getGrad();
    ASSERT_TRUE(xgrad == base::Tensor<f64>(base::Shape({2, 3, 3}), {
        478.756440000, 521.930520000, 565.104600000,
        800.181720000, 869.690520000, 939.199320000,
        1067.947560000, 1158.458760000, 1248.969960000,
        376.688280000, 412.719960000, 448.751640000,
        648.119640000, 710.486040000, 772.852440000,
        872.915400000, 956.284200000, 1039.653000000,
    }));
    c.clearGrad();
    c.createGradGraph();
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1->getGradGraph());
    c.clearGradGraph();
    ASSERT_TRUE(cx1.call() == xgrad);
    cx1.deriv();
    ASSERT_TRUE(x1->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 3}), {
        5313.495600000, 5735.883600000, 6158.271600000,
        7938.414000000, 8512.304400000, 9086.194800000,
        10122.382800000, 10819.299600000, 11516.216400000,
        4398.322800000, 4768.237200000, 5138.151600000,
        6727.839600000, 7249.256400000, 7770.673200000,
        8660.917200000, 9305.360400000, 9949.803600000,
    }));
    cx1.clearGrad();
}

TEST(Test_grad_tensor2, test) {
    // 测试：m1 = [[1, 2], [3, 4]], m2 = [[2, 3, 4], [5, 6, 7]]
    // f(m1, m2) = Transpose(inv(Transpose(m1*m2) * m2) * Transpose([1,1,1])) * Transpose([1, 1, 1])
    auto t1 = base::Tensor<f64>(base::Shape({2, 2, 2}), {1, 2, 3, 4, 1, 2, 2, 1});
    auto t2 = base::Tensor<f64>(base::Shape({2, 3, 2}), {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1});
    auto t3 = base::Tensor<f64>(base::Shape({2, 1, 3}), {1, 1, 1, 1, 1, 1});
    auto x1_ = std::make_shared<op::DataOp<base::Tensor<f64>>>(t1, true);
    auto x1 = std::make_shared<op::Inv<base::Tensor<f64>>>(x1_);
    auto x2_ = std::make_shared<op::DataOp<base::Tensor<f64>>>(t2, true);
    auto x2 = std::make_shared<op::Reshape<base::Tensor<f64>, base::Shape>>(x2_, base::Shape({2, 2, 3}));
    auto ct = std::make_shared<op::DataOp<base::Tensor<f64>>>(t3);
    auto item1 = std::make_shared<op::Mmul<base::Tensor<f64>>>(x1, x2);
    auto item2 = std::make_shared<op::Transpose<base::Tensor<f64>>>(item1, -2, -1);
    auto item3 = std::make_shared<op::Mmul<base::Tensor<f64>>>(item2, x2);
    auto item = std::make_shared<op::Sum<base::Tensor<f64>, base::Shape>>(item3);
    auto c = calc::Calculator<base::Tensor<f64>>(item);
    ASSERT_TRUE(c.call() == base::Tensor<f64>(base::Shape({1}), {73.5}));
    c.deriv();
    ASSERT_TRUE(x1_->getGrad() == base::Tensor<f64>(base::Shape({2, 2, 2}), {-31.5, -15.75, 4.5, 2.25, -1, 8, 8, -64}));
    ASSERT_TRUE(x2_->getGrad() == base::Tensor<f64>(base::Shape({2, 3, 2}), {13.5, 13.5, 13.5, 0, 0, 0, -2, -2, -2, 16, 16, 16}));
    c.clearGrad();
    c.createGradGraph();
    auto cx1 = calc::Calculator<base::Tensor<f64>>(x1_->getGradGraph());
    auto cx2 = calc::Calculator<base::Tensor<f64>>(x2_->getGradGraph());
    c.clearGradGraph();
    ASSERT_TRUE(cx1.call() == base::Tensor<f64>(base::Shape({2, 2, 2}), {-31.5, -15.75, 4.5, 2.25, -1, 8, 8, -64}));
    ASSERT_TRUE(cx2.call() == base::Tensor<f64>(base::Shape({2, 3, 2}), {13.5, 13.5, 13.5, 0, 0, 0, -2, -2, -2, 16, 16, 16}));
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
    ASSERT_TRUE(t2.sum(0).expand(0, 2) == base::Tensor<f64>(base::Shape({2, 2, 3}), {2, 4, 6, 8, 10, 12, 2, 4, 6, 8, 10, 12}));
    ASSERT_TRUE(t2.sum(-1).expand(-1, 3) == base::Tensor<f64>(base::Shape({2, 2, 3}), {6, 6, 6, 15, 15, 15, 6, 6, 6, 15, 15, 15}));
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
    base::Tensor<f64> t8 = base::Tensor<f64>(base::Shape({2, 2, 1}), {1, 2, 3, 4});
    base::Tensor<f64> t9 = base::Tensor<f64>(base::Shape({1, 2, 3}), {7, 8, 9, 10, 11, 12});
    base::Tensor<f64> s1, s2;
    ASSERT_TRUE(t2.cat(t8, -1) == base::Tensor<f64>(base::Shape({2, 2, 4}), {1, 2, 3, 1, 4, 5, 6, 2, 1, 2, 3, 3, 4, 5, 6, 4}));
    t2.cat(t8, -1).split(-1, 3, s1, s2);
    ASSERT_TRUE(s1 == t2);
    ASSERT_TRUE(s2 == t8);
    ASSERT_TRUE(t2.cat(t9, 0) == base::Tensor<f64>(base::Shape({3, 2, 3}), {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    t2.cat(t9, 0).split(0, 2, s1, s2);
    ASSERT_TRUE(s1 == t2);
    ASSERT_TRUE(s2 == t9);
    // 测试多个cat/split
    base::Tensor<f64> t10 = base::Tensor<f64>(base::Shape({1, 2}), {1, 2});
    base::Tensor<f64> t11 = base::Tensor<f64>(base::Shape({2, 2}), {3, 4, 5, 6});
    base::Tensor<f64> t12 = base::Tensor<f64>(base::Shape({1, 2}), {7, 8});
    ASSERT_TRUE(base::Tensor<f64>::cat({t10, t11, t12}, 0) == base::Tensor<f64>(base::Shape({4, 2}), {1, 2, 3, 4, 5, 6, 7, 8}));
    auto l = base::Tensor<f64>::split(base::Tensor<f64>::cat({t10, t11, t12}, 0), {1, 2}, 0);
    ASSERT_TRUE(l.size() == 3);
    ASSERT_TRUE(l[0] == base::Tensor<f64>(base::Shape({1, 2}), {1, 2}));
    ASSERT_TRUE(l[1] == base::Tensor<f64>(base::Shape({2, 2}), {3, 4, 5, 6}));
    ASSERT_TRUE(l[2] == base::Tensor<f64>(base::Shape({1, 2}), {7, 8}));
}

TEST(Test_grad_descent, test) {
    auto t1 = base::Tensor<f64>(base::Shape({2}), {1, 2});
    auto x = std::make_shared<op::DataOp<base::Tensor<f64>>>(t1, true);

    auto cost_func = [](const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>& args) {
        auto t2 = base::Tensor<f64>(base::Shape({2}), {2, 3});
        auto t3 = base::Tensor<f64>(base::Shape({2}), {4, 5});
        auto ct1 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t2);
        auto ct2 = std::make_shared<op::DataOp<base::Tensor<f64>>>(t3);
        auto item1 = std::make_shared<op::Mul<base::Tensor<f64>>>(args[0], args[0]);
        auto item2 = std::make_shared<op::Mul<base::Tensor<f64>>>(ct1, args[0]);
        auto item3 = std::make_shared<op::Add<base::Tensor<f64>>>(item1, item2);
        auto item4 = std::make_shared<op::Add<base::Tensor<f64>>>(item3, ct2);
        auto item5 = std::make_shared<op::Reshape<base::Tensor<f64>, base::Shape>>(item4, base::Shape({1, 2}));
        auto item6 = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({2, 1}), 1));
        auto item = std::make_shared<op::Mmul<base::Tensor<f64>>>(item5, item6);
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

    auto d2 = std::make_shared<algo::Optimizer>(cost_func, std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>{x}, true);
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