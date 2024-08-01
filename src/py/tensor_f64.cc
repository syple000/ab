// 对python仅暴露op，并将op作为数据
// 在每一次数据读取时进行计算

#include "auto_engine/algo/opt.h"
#include "auto_engine/base/basic_types.h"
#include "auto_engine/calc/calc.h"
#include "auto_engine/cuda/info.h"
#include "auto_engine/op/add.h"
#include "auto_engine/op/add_n.h"
#include "auto_engine/op/data_op.h"
#include "auto_engine/op/div.h"
#include "auto_engine/op/div_n.h"
#include "auto_engine/op/inv.h"
#include "auto_engine/op/log.h"
#include "auto_engine/op/mmul.h"
#include "auto_engine/op/mul.h"
#include "auto_engine/op/mul_n.h"
#include "auto_engine/op/op.h"
#include "auto_engine/op/pow.h"
#include "auto_engine/op/pow_n.h"
#include "auto_engine/op/reshape.h"
#include "auto_engine/op/sin_cos.h"
#include "auto_engine/op/sub.h"
#include "auto_engine/op/sub_n.h"
#include "auto_engine/op/sum_d_expand_d.h"
#include "auto_engine/op/sum_expand.h"
#include "auto_engine/shape/shape.h"
#include "auto_engine/tensor/tensor.h"
#include "pybind11/pybind11.h"
#include "fmt/core.h"
#include <functional>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// 性能一般，c++性能会更好

void init() {
    google::InitGoogleLogging("ae");
    google::SetLogDestination(google::INFO, "./info.log");
    google::SetLogDestination(google::ERROR, "./error.log");
    google::SetLogDestination(google::WARNING, "./warning.log");
    google::SetStderrLogging(google::WARNING);

    cuda::init();
}

PYBIND11_MODULE(ae, m) {
    init();

    m.doc() = "auto engine";

    py::register_exception<std::invalid_argument>(m, "invalid_argument");
    py::register_exception<std::runtime_error>(m, "runtime_err");

    py::class_<op::Op<base::Tensor<f64>>, std::shared_ptr<op::Op<base::Tensor<f64>>>>(m, "op")
        .def("__repr__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::string {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.call();
            return fmt::format("{{\"shape\": {0}, \"data\": {1}}}", op->getOutput().shape().toString(), op->getOutput().toString(true));
        })
        .def("update", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) {
            calc::Calculator<base::Tensor<f64>> c(op2);
            c.call();
            op1->setOutput(op2->getOutput());
        })
        .def("item", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> f64 {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.call();
            auto data = op->getOutput().data();
            if (data.size() != 1) {
                throw std::runtime_error(fmt::format("tensor size = {}", data.size()));
            }
            return data[0];
        })
        .def("shape", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> const std::vector<u32>& {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.call();
            return op->getOutput().shape().getDims();
        })
        .def("tolist", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> py::list {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.call();
            auto data = op->getOutput();
            if (data.shape().dimCnt() == 0) {return py::none();}

            int data_index = 0;
            std::function<py::list(int dim_index)> f = [&data, &data_index, &f](int dim_index) -> py::list {
                if (dim_index == data.shape().dimCnt() - 1) {
                    py::list lst(data.shape().getDim(dim_index));
                    for (int i = 0; i < data.shape().getDim(dim_index); i++) {
                        lst[i] = data.data()[data_index++];
                    }
                    return lst;
                }
                py::list lst(data.shape().getDim(dim_index));
                for (int i = 0; i < data.shape().getDim(dim_index); i++) {
                    lst[i] = f(dim_index + 1);
                }
                return lst;
            };
            return f(0);
        })
        .def("grad", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::DataOp<base::Tensor<f64>>>(op->getGrad(), false);
        })
        .def("grad_graph", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return op->getGradGraph();
        })
        .def("backward", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.deriv();
        })
        .def("create_grad_graph", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.createGradGraph();
        })
        .def("clear_output", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.clearOutput();
        })
        .def("clear_grad", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.clearGrad();
        })
        .def("clear_grad_graph", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) {
            calc::Calculator<base::Tensor<f64>> c(op);
            c.clearGradGraph();
        })
        // 运算
        .def("__add__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Add<base::Tensor<f64>>>(op1, op2);
        })
        .def("__add__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, f64 n) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            auto add_op = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({1}), n));
            return std::make_shared<op::AddN<base::Tensor<f64>, base::Shape>>(op, add_op);
        })
        .def("add_n", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::AddN<base::Tensor<f64>, base::Shape>>(op1, op2);
        })
        .def("__sub__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Sub<base::Tensor<f64>>>(op1, op2);
        })
        .def("__sub__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, f64 n) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            auto sub_op = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({1}), n));
            return std::make_shared<op::SubN<base::Tensor<f64>, base::Shape>>(op, sub_op);
        })
        .def("sub_n", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::SubN<base::Tensor<f64>, base::Shape>>(op1, op2);
        })
        .def("__mul__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Mul<base::Tensor<f64>>>(op1, op2);
        })
        .def("__mul__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, f64 n) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            auto mul_op = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({1}), n));
            return std::make_shared<op::MulN<base::Tensor<f64>, base::Shape>>(op, mul_op);
        })
        .def("mul_n", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::MulN<base::Tensor<f64>, base::Shape>>(op1, op2);
        })
        .def("__truediv__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Div<base::Tensor<f64>>>(op1, op2);
        })
        .def("__truediv__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, f64 n) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            auto div_op = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({1}), n));
            return std::make_shared<op::DivN<base::Tensor<f64>, base::Shape>>(op, div_op);
        })
        .def("div_n", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::DivN<base::Tensor<f64>, base::Shape>>(op1, op2);
        })
        .def("__pow__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Pow<base::Tensor<f64>>>(op1, op2);
        })
        .def("__pow__", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, f64 n) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            auto pow_op = std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(base::Shape({1}), n));
            return std::make_shared<op::PowN<base::Tensor<f64>, base::Shape>>(op, pow_op);
        })
        .def("pow_n", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::PowN<base::Tensor<f64>, base::Shape>>(op1, op2);
        })
        .def("sin", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Sin<base::Tensor<f64>>>(op);
        })
        .def("cos", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Cos<base::Tensor<f64>>>(op);
        })
        .def("log", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Log<base::Tensor<f64>>>(op);
        })
        .def("sum", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Sum<base::Tensor<f64>, base::Shape>>(op);
        })
        .def("expand", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, const std::vector<u32>& dims) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Expand<base::Tensor<f64>, base::Shape>>(op, base::Shape(dims));
        })
        .def("sum", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, int d) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::SumD<base::Tensor<f64>, base::Shape>>(op, d);
        })
        .def("expand", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, const std::vector<u32>& dims, int d) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::ExpandD<base::Tensor<f64>, base::Shape>>(op, base::Shape(dims), d);
        })
        .def("mm", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op1, std::shared_ptr<op::Op<base::Tensor<f64>>> op2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Mmul<base::Tensor<f64>>>(op1, op2);
        })
        .def("transpose", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, int d1, int d2) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Transpose<base::Tensor<f64>>>(op, d1, d2);
        }, py::arg("d1")=-2, py::arg("d2")=-1)
        .def("inverse", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Inv<base::Tensor<f64>>>(op);
        })
        .def("reshape", [](std::shared_ptr<op::Op<base::Tensor<f64>>> op, const std::vector<u32>& dims) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
            return std::make_shared<op::Reshape<base::Tensor<f64>, base::Shape>>(op, base::Shape(dims));
        });

    py::class_<algo::Optimizer, std::shared_ptr<algo::Optimizer>>(m, "opt_algo")
        .def(py::init<std::function<std::shared_ptr<op::Op<base::Tensor<f64>>>(const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>&)>,
            const std::vector<std::shared_ptr<op::Op<base::Tensor<f64>>>>&,
            bool>(),
            py::arg("cost_func"), py::arg("vars"), py::arg("fix_cost_graph") = false
        )
        .def("algo_hyper_params", &algo::Optimizer::algoHyperParams, py::arg("algo"), py::arg("hyper_params"))
        .def("run", &algo::Optimizer::run);


    m.def("tensor", [](py::list lst, bool requires_grad=false) -> std::shared_ptr<op::Op<base::Tensor<f64>>> {
        std::vector<u32> dims;
        std::vector<f64> data;
        std::function<void(py::list, int)> f = [&dims, &data, &f](py::list lst, int level) {
            if (dims.size() == level) {
                dims.push_back(lst.size());
            }
            if (dims[level] != lst.size()) {
                throw std::invalid_argument(fmt::format("level {}: {}, {} conflict", level, dims[level], lst.size()));
            }
            for (auto e : lst) {
                if (py::isinstance<py::list>(e)) {
                    auto elst = py::cast<py::list>(e);
                    f(elst, level + 1);       
                } else if (py::isinstance<py::int_>(e)) {
                    auto ei = py::cast<i64>(e);
                    data.push_back(ei);
                } else if (py::isinstance<py::float_>(e)) {
                    auto ef = py::cast<f64>(e);
                    data.push_back(ef);
                } else {
                    throw std::invalid_argument(fmt::format("unknown input type: level: {}", level));
                }
            }       
        };
        f(lst, 0);
        auto shape = base::Shape(dims);
        if (shape.tensorSize() != data.size()) {
            throw std::invalid_argument("not a tensor");
        }
        return std::make_shared<op::DataOp<base::Tensor<f64>>>(base::Tensor<f64>(shape, data), requires_grad);
    }, py::arg("lst"), py::arg("requires_grad")=false);


}