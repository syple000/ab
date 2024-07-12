
#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/slice.h"
#include "auto_engine/op/bop.h"
#include "auto_engine/op/add.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/div.h"
#include "auto_engine/tensor/shape.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/pow.h"
#include "auto_engine/op/log.h"
#include "auto_engine/op/sin_cos.h"
#include "auto_engine/tensor/tensor.h"
#include "auto_engine/cuda/matrix_f64.h"
#include "gtest/gtest.h"
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

TEST(Test_f64, Test){
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

TEST(Test_tensor, Test){
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

TEST(Cuda, Cuda) {
    std::vector<f64> vec{1, 2, 3, 4, 5, 6};
    base::Slice<f64> slice(&vec, 0, 6);
    cuda::MatrixF64 m1(2, 3, slice);
    ASSERT_TRUE(m1 == cuda::MatrixF64(2, 3, {1, 2, 3, 4, 5, 6}));
    auto m2 = m1.transpose();
    ASSERT_TRUE(m2 == cuda::MatrixF64(3, 2, {1, 4, 2, 5, 3, 6}));
    cuda::MatrixF64 m3(3, 2, slice);
    ASSERT_TRUE(m3 == cuda::MatrixF64(3, 2, {1, 2, 3, 4, 5, 6}));
    auto m4 = m1.mmul(m3);
    ASSERT_TRUE(m4 == cuda::MatrixF64(2, 2, {22, 28, 49, 64}));
    auto m5 = m3.mmul(m1);
    ASSERT_TRUE(m5 == cuda::MatrixF64(3, 3, {9, 12, 15, 19, 26, 33, 29, 40, 51}));
    auto m6 = cuda::MatrixF64(3, 3, {1, 2, 3, 4, 5, 6, 7, 2, 9});
    auto m7 = m6.inv();
    ASSERT_TRUE(m7 == cuda::MatrixF64(3, 3, {-11.0/12, 1.0/3, 1.0/12, -1.0/6, 1.0/3, -1.0/6, 3.0/4, -1.0/3, 1.0/12}));
    auto m8 = cuda::MatrixF64(3, 3, {0, 2, 3, 4, 5, 6, 7, 2, 9});
    auto m9 = m8.inv();
    ASSERT_TRUE(m9 == cuda::MatrixF64(3, 3, {-11.0/23, 4.0/23, 1.0/23, -2.0/23, 7.0/23, -4.0/23, 9.0/23, -14.0/69, 8.0/69}));
    auto m10 = cuda::MatrixF64(2, 3, {1, 2, 3, 4, 5, 6});
    auto m11 = m10.sin();
    ASSERT_TRUE(m11 == cuda::MatrixF64(2, 3, {sin(1), sin(2), sin(3), sin(4), sin(5), sin(6)}));
    auto m12 = m10.add(m10);
    ASSERT_TRUE(m12 == cuda::MatrixF64(2, 3, {2, 4, 6, 8, 10, 12}));
    auto m13 = m10.add(m12);
    ASSERT_TRUE(m13 == cuda::MatrixF64(2, 3, {3, 6, 9, 12, 15, 18}));
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "./info.log");
    google::SetLogDestination(google::ERROR, "./error.log");
    google::SetLogDestination(google::WARNING, "./warning.log");

    ::testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();

    google::ShutdownGoogleLogging();
    return 0;
}