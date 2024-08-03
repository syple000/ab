#ifndef OP_LINK_OP_H
#define OP_LINK_OP_H

#include "op.h"
#include <memory>

namespace op {

// 仅链接op。前向传播的数据由子节点赋值；反向传播的数据由下一层节点赋值，子节点的反向传播由子节点自行计算

template<typename T>
class LinkOp: public Op<T> {
protected:
    LinkOp(std::shared_ptr<Op<T>> arg) : Op<T>(arg) {}
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg) {
        return std::shared_ptr<LinkOp<T>>(new LinkOp<T>(arg));
    }
    virtual void forward() override {};
    virtual void backward() override {};
    virtual void createGradGraph() override {};
    virtual std::string name() const override {return "Link";}
};

}

#endif