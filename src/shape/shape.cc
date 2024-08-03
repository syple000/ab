#include "auto_engine/shape/shape.h"
#include "auto_engine/base/exit_code.h"
#include "auto_engine/config/config.h"
#include "glog/logging.h"
#include <cstdlib>
#include <fmt/core.h>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace base {

void Shape::calcStrides() {
    if (_dims.size() == 0) {
        return;
    }
    _strides = std::vector<u32>(_dims.size());
    _strides[_dims.size()-1] = 1;
    for (int i = _dims.size() - 2; i >= 0; i--) {
        _strides[i] = _strides[i+1] * _dims[i+1];
        if (_strides[i] < _strides[i+1] || _strides[i] < _dims[i+1]) {
            LOG(ERROR) << __FUNCTION__ << "shape size overflow";
            exit(SHAPE_SIZE_OVERFLOW);
        }
    }
    _size = _strides[0] * _dims[0];
    if (_size < _strides[0] || _size < _dims[0]) {
        LOG(ERROR) << __FUNCTION__ << "shape size overflow";
        exit(SHAPE_SIZE_OVERFLOW);
    }
}

Shape::Shape(const std::vector<u32>& dims) {
    if (dims.size() > MAX_TENSOR_DIM_CNT) {
        LOG(ERROR) << __FUNCTION__ << " dims size out of range";
        exit(SHAPE_DIM_SIZE_LIMIT);
    }
    _dims = dims;
    calcStrides();
}

Shape::Shape(const Shape& shape) {
    _dims = shape._dims;
    _strides = shape._strides;
    _size = shape._size;
}

Shape::Shape(Shape&& shape) {
    _dims = std::move(shape._dims);
    _strides = std::move(shape._strides);
    _size = shape._size;
}

Shape& Shape::operator=(const Shape& shape) {
    if (this == &shape) {
        return *this;
    } 
    _dims = shape._dims;
    _strides = shape._strides;
    _size = shape._size;
    return *this;
}

Shape& Shape::operator=(Shape&& shape) {
    if (this == &shape) {
        return *this;
    } 
    _dims = std::move(shape._dims);
    _strides = std::move(shape._strides);
    _size = shape._size;
    return *this;
}

bool Shape::operator==(const Shape& shape) const {
    return _dims == shape._dims;
}

bool Shape::reshape(const std::vector<u32>& dims, Shape& shape) const {
    shape = Shape(dims);
    if (_size != shape._size) {
        LOG(ERROR) << "check reshapeable fail";
        return false;
    }
    return true;
}

std::string Shape::toString() const {
    std::stringstream stream;
    stream << "(";
    for (int i = 0; i < _dims.size(); i++) {
        if (i != 0) {
            stream << ", ";
        }
        stream << std::to_string(_dims[i]);
    }
    stream << ")";
    return stream.str();
}

bool Shape::sum(Shape& shape) const { // 必须至少有一个元素返回
    if (_size == 0) {
        LOG(ERROR) << "sum null tensor";
        return false;
    }
    shape = Shape({1});
    return true;
}

bool Shape::sum(int d, Shape& shape) const {
    if (d < 0 || d >= _dims.size()) {
        LOG(ERROR) << fmt::format("sum dim out of range: {}, shape: {}", d, _dims.size());
        return false;
    }
    std::vector<u32> dims;
    dims.reserve(_dims.size() - 1);
    for (u32 i = 0; i < _dims.size(); i++) {
        if (i == d) {continue;}
        dims.emplace_back(_dims[i]);
    }
    shape = Shape(dims);
    return true;
}


bool Shape::expand(const Shape& shape, int d) const {
    if (shape.tensorSize() <= 0) {
        LOG(ERROR) << "expand target shape 0";
        return false;
    }
    if (d < 0 || d >= shape._dims.size()) {
        LOG(ERROR) << fmt::format("expand d: {} out of range: {}", d, shape._dims.size());
        return false;
    }
    if (shape._dims.size() != _dims.size() + 1) {
        LOG(ERROR) << fmt::format("expand fail due to size unmatch, shape: {}, org shape: {}, d: {}", shape.toString(), toString(), d);
        return false;
    }
    u32 idiff = 0;
    for (u32 i = 0; i < _dims.size(); i++) {
        if (i == d) {
            idiff += 1;
        }
        if (_dims[i] != shape._dims[i + idiff]) {
            LOG(ERROR) << fmt::format("expand fail due to dim unmatch, shape: {}, org shape: {}, d: {}", shape.toString(), toString(), d);
            return false;
        }
    }

    return true;
}

bool Shape::expand(const Shape& shape) const {
    if (_size != 1) {
        LOG(ERROR) << fmt::format("expand dim err tensor size ne 1: {}", _size);
        return false;
    }
    if (shape.tensorSize() <= 0) {
        LOG(ERROR) << "expand dim err target size eq 0";
        return false;
    }
    return true;
}

bool Shape::cat(const std::vector<std::reference_wrapper<Shape>>& ss, u32 d, Shape& shape) {
    if (ss.size() == 0) {
        LOG(ERROR) << "cat tensor list size 0";
        return false;
    }
    auto s = ss[0].get();
    if (d >= s.dimCnt()) {
        LOG(ERROR) << fmt::format("cat tensor d: {} out of range: {}", d, s.dimCnt());
        return false;
    }
    auto dims = s.getDims();
    for (u32 i = 1; i < ss.size(); i++) {
        auto e = ss[i].get();
        if (e.dimCnt() != s.dimCnt()) {
            LOG(ERROR) << fmt::format("cat {}th tensor dim cnt: {} not matched, target: {}", i, e.dimCnt(), s.dimCnt());
            return false;
        }
        for (u32 j = 0; j < s.dimCnt(); j++) {
            if (j == d) {dims[j] += e.getDim(j); continue;}
            if (s.getDim(j) != e.getDim(j)) {
                LOG(ERROR) << fmt::format("cat {}th tensor {}th dim not matched", i, j);
                return false;
            }
        }
    }
    shape = Shape(dims);
    return true;
}


bool Shape::cat(const Shape& dst, u32 d, u32 d_offset) {
    if (d >= dst.dimCnt()) {
        LOG(ERROR) << fmt::format("cat d: {} out of range: {}", d, dst.dimCnt());
        return false;
    }
    if (dimCnt() != dst.dimCnt()) {
        LOG(ERROR) << fmt::format("cat dim cnt ne: {}, {}", dimCnt(), dst.dimCnt());
        return false;
    }
    for (u32 i = 0; i < dst.dimCnt(); i++) {
        if (i == d) {
            if (d_offset + getDim(d) > dst.getDim(d)) {
                LOG(ERROR) << fmt::format("cat d: {} fail, dim limit", d);
                return false;
            }
        } else {
            if (getDim(i) != dst.getDim(i)) {
                LOG(ERROR) << fmt::format("split dim: {} not eq: {}, {}", i, getDim(i), dst.getDim(i));
                return false;
            }               
        }
    }
    return true;
}

bool Shape::split(const Shape& shape, const std::vector<u32>& sl, u32 d, std::vector<Shape>& ss) {
    if (d >= shape.dimCnt()) {
        LOG(ERROR) << fmt::format("split tensor d: {} out of range: {}", d, shape.dimCnt());
        return false;
    }
    auto dims = shape.getDims();
    ss = std::vector<Shape>(sl.size());
    for (u32 i = 0; i < sl.size(); i++) {
        if (dims[d] >= sl[i]) {
            dims[d] -= sl[i];
        } else {
            LOG(ERROR) << fmt::format("split fail: {} due to neg", d);
            return false;
        }
        auto sd = dims; sd[d] = sl[i];
        ss[i] = Shape(sd);
    }
    return true;
}

bool Shape::split(const Shape& dst, u32 d, u32 d_offset) {
    if (d >= dst.dimCnt()) {
        LOG(ERROR) << fmt::format("split d: {} out of range: {}", d, dst.dimCnt());
        return false;
    }
    if (dimCnt() != dst.dimCnt()) {
        LOG(ERROR) << fmt::format("split dim cnt ne: {}, {}", dimCnt(), dst.dimCnt());
        return false;
    }
    for (u32 i = 0; i < dst.dimCnt(); i++) {
        if (i == d) {
            if (d_offset + dst.getDim(d) > getDim(d)) {
                LOG(ERROR) << fmt::format("split d: {} fail, dim limit", d);
                return false;
            }
        } else {
            if (getDim(i) != dst.getDim(i)) {
                LOG(ERROR) << fmt::format("split dim: {} not eq: {}, {}", i, getDim(i), dst.getDim(i));
                return false;
            }               
        }
    }
    return true;
}

bool Shape::permute(const std::vector<u32>& pl, Shape& shape) const {
    if (pl.size() != _dims.size()) {
        LOG(ERROR) << fmt::format("permute pl len != dim cnt: {}, {}", pl.size(), _dims.size());
        return false;
    }
    std::vector<u32> dims(pl.size());
    std::unordered_set<u32> pl_set; pl_set.reserve(pl.size());
    for (u32 i = 0; i < pl.size(); i++) {
        if (pl[i] >= _dims.size()) {
            LOG(ERROR) << fmt::format("permute index out of range: {}, {}", i, pl[i]);
            return false;
        }
        if (pl_set.find(pl[i]) != pl_set.end()) {
            LOG(ERROR) << fmt::format("permute index dup: {}, {}", i, pl[i]);
            return false;
        }
        pl_set.insert(pl[i]);
        dims[i] = _dims[pl[i]];
    }
    shape = Shape(dims);
    return true;
}

bool Shape::transpose(int d1, int d2, Shape& shape) const {
    if (d1 < 0 || d2 < 0 || d1 == d2 || d1 >= _dims.size() || d2 >= _dims.size()) {
        LOG(ERROR) << fmt::format("transpose d1/d2 invalid: {}, {}. shape dim cnt: {}", d1, d2, _dims.size());
        return false;
    }
    auto dims = _dims;
    std::swap(dims[d1], dims[d2]);
    shape =  Shape(dims);
    return true;
}

bool Shape::mmul(const Shape& s, Shape& shape) const {
    if (_dims.size() < 2 || s._dims.size() < 2) {
        LOG(ERROR) << "dim cnt lt 2";
        return false;
    }
    if (_dims[_dims.size() - 1] != s._dims[s._dims.size() - 2]) {
        LOG(ERROR) << "can not mul, col cnt != row cnt";
        return false;
    }
    std::vector<u32> dims1(_dims.begin(), _dims.end() - 2);
    std::vector<u32> dims2(s._dims.begin(), s._dims.end() - 2);
    if (dims1 != dims2) {
        LOG(ERROR) << "matrix not equal";
        return false;
    }

    dims1.push_back(_dims[_dims.size() - 2]);
    dims1.push_back(s._dims[s._dims.size() - 1]);
    shape =  Shape(dims1);
    return true;
}

}