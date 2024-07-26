#ifndef OP_OP_H
#define OP_OP_H

#include "auto_engine/base/basic_types.h"
#include "auto_engine/base/exit_code.h"
#include "auto_engine/utils/vec_utils.h"
#include "glog/logging.h"
#include <cstdlib>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace op {

template<typename T> // T必须支持四则运算
class Op : public std::enable_shared_from_this<Op<T>> {
public:
    template<typename... Args>
    Op(Args... args) {
        _args = convertToVec(args...);
        for (auto arg : _args) {
            if (arg->getRequiresGrad()) {
                setRequiresGrad(true);
                break;
            }
        }
    }
    Op(const T& data, bool requires_grad) {
        setOutput(data);
        setRequiresGrad(requires_grad);
    }
    virtual ~Op() {
        for (auto arg : this->_args) {
            arg->template clearGradGraph();
        }
        if (_gen_exec_queue) {
            _exec_queue.clear();
        }
    };
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void createGradGraph() = 0;
    virtual std::string name() const = 0;

    const T& getOutput() const {
        if (!_has_output) {
            LOG(ERROR) << "[" << __FUNCTION__ << "]" << "touch unset output: ";
            exit(TOUCH_UNSET_OUTPUT);
        }
        return _output;
    }
    void setOutput(const T& data) {_output = data; _has_output = true;}
    void setOutput(T&& data) {_output = std::move(data); _has_output = true;}
    bool hasOutput() const {return _has_output;}
    void clearOutput() {_has_output = false;}

    const T& getGrad() const {
        if (!_has_grad) {
            LOG(ERROR) << "[" << __FUNCTION__ << "]" << "touch unset grad: ";
            exit(TOUCH_UNSET_GRAD);
        }
        return _grad;
    }
    void setGrad(const T& data) {_grad = data; _has_grad = true;}
    void setGrad(T&& data) {_grad = std::move(data); _has_grad = true;}
    bool hasGrad() const {return _has_grad;}
    void clearGrad() {_has_grad = false;} // 梯度累计求和计算，每次计算开始需要保证梯度清空

    std::shared_ptr<Op<T>> getGradGraph() const {
        if (!_grad_graph) {
            LOG(ERROR) << "[" << __FUNCTION__ << "]" << "touch null grad graph: ";
            exit(TOUCH_UNSET_GRAD_GRAPH); 
        }
        return _grad_graph;
    }
    void setGradGraph(std::shared_ptr<Op<T>> graph) {_grad_graph = graph;}
    bool hasGradGraph() const {return _grad_graph != nullptr;}
    void clearGradGraph() {_grad_graph = nullptr;} // 同梯度，会累计求和更新；并且，如果不清空会内存循环引用，引起内存泄漏

    bool getRequiresGrad() const {return _requires_grad;}
    void setRequiresGrad(bool requires_grad) {_requires_grad = requires_grad;}

    const std::vector<std::shared_ptr<Op<T>>>& args() const {return _args;}
    std::shared_ptr<Op<T>> arg() const {return _args[0];}
    std::shared_ptr<Op<T>> arg1() const {return _args[0];}
    std::shared_ptr<Op<T>> arg2() const {return _args[1];}

    const std::vector<std::shared_ptr<Op<T>>>& exec_queue() {
        if (_gen_exec_queue) {
            return _exec_queue;
        }
        _gen_exec_queue = true;

        std::unordered_set<std::shared_ptr<op::Op<T>>> visted;
        std::function<void(std::shared_ptr<op::Op<T>>)> recur = [&recur, &visted, this](std::shared_ptr<op::Op<T>> op) {
            if (visted.find(op) != visted.end()) {
                return;
            }
            visted.insert(op);
            for (auto sop : op->template args()) {
                recur(sop);
            }
            if (op != this->template shared_from_this()) {
                _exec_queue.push_back(op);
            }
        };
        recur(this->template shared_from_this());
        return _exec_queue;
    }
private: // 继承类仅关注方法，不直接操作数据
    T _output;
    bool _has_output = false;
    T _grad;
    bool _has_grad = false;
    std::shared_ptr<Op<T>> _grad_graph;
    bool _requires_grad = false;
    std::vector<std::shared_ptr<Op<T>>> _args;

    // 执行图，懒执行
    bool _gen_exec_queue = false;
    std::vector<std::shared_ptr<Op<T>>> _exec_queue;
};

template<typename T, typename... Args>
class OpFunc {
public:
    virtual T call(const Args&...) = 0;
    // 导数计算需要指定参数位置，传入上层已计算梯度，和本次计算的参数
    virtual T deriv(u32, const T&, const Args&...) = 0;
    virtual std::shared_ptr<Op<T>> derivFunc(u32, std::shared_ptr<Op<T>>, std::shared_ptr<Op<Args>>...) = 0;
};

template<int N, typename T, typename... Args>
struct GenOpFunc {
    using type = typename GenOpFunc<N - 1, T, T, Args...>::type;
};
template<typename T, typename... Args>
struct GenOpFunc<0, T, Args...> {
    using type = OpFunc<T, Args...>; 
};

template<int N, typename T>
using GenOpFuncT = typename GenOpFunc<N, T>::type;

}



#endif