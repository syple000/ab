#ifndef OP_PERMUTE_H
#define OP_PERMUTE_H

#include "auto_engine/op/methods.h"
#include "auto_engine/op/uop.h"
#include <memory>
#include <vector>
namespace op {

template<typename T>
class Permute: public UOP<T> {
protected:
    Permute(std::shared_ptr<Op<T>> arg, const std::vector<u32>& pl): UOP<T>(arg), _pl(pl), _npl(pl.size()) {
        for (u32 i = 0; i < _pl.size(); i++) {
            _npl[_pl[i]] = i;
        }
    }
public:
    static std::shared_ptr<Op<T>> op(std::shared_ptr<Op<T>> arg, const std::vector<u32>& pl) {
        auto op = std::shared_ptr<Permute<T>>(new Permute<T>(arg, pl));
        op->template forward();
        return op;
    }

    T call(const T& arg) override {
        return permute(arg, _pl);
    }

    T deriv(u32 _, const T& grad, const T& arg) override {
        return permute(grad, _npl);
    }

    std::shared_ptr<Op<T>> derivFunc(u32 _, std::shared_ptr<Op<T>> grad, std::shared_ptr<Op<T>> arg) override {
        return Permute<T>::op(grad, _npl);
    }

    std::string name() const override {return "Permute";}
private:
    std::vector<u32> _pl, _npl;
};

}

#endif