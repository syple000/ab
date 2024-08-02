#ifndef OP_DATA_OP_H
#define OP_DATA_OP_H

#include "op.h"
#include <memory>

namespace op {

template<typename T>
class DataOp: public Op<T> {
protected:
    DataOp(const T& data, bool requires_grad = false) : Op<T>(data, requires_grad) {}
public:
    static std::shared_ptr<Op<T>> op(const T& data, bool requires_grad = false) {
        return std::shared_ptr<DataOp<T>>(new DataOp<T>(data, requires_grad));
    }
    virtual void forward() override {};
    virtual void backward() override {};
    virtual void createGradGraph() override {};
    virtual std::string name() const override {return "Data";}
};

}

#endif