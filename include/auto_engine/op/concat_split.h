#ifndef OP_CONCAT_SPLIT_H
#define OP_CONCAT_SPLIT_H

#include "auto_engine/op/vaop.h"
namespace op {

template<typename T, typename SHAPE>
class Concat: public VAOP<T> {
public:
    T call(const std::vector<std::reference_wrapper<T>>&) override;
    T deriv(u32, const T&, const std::vector<std::reference_wrapper<T>>&) override;
    std::shared_ptr<Op<T>> derivFunc(u32, std::shared_ptr<Op<T>>, const std::vector<std::shared_ptr<Op<T>>>&) override;
    std::string name() const override {return "Concat";}
};

}

#endif