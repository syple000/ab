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

Shape Shape::reshape(const std::vector<u32>& dims) const {
    auto shape = Shape(dims);
    if (_size != shape._size) {
        LOG(ERROR) << "check reshapeable fail";
        return Shape();
    }
    return Shape(dims);
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

Shape Shape::sum() const {
    if (_size == 0) {
        LOG(ERROR) << "sum null tensor";
        return Shape();
    }
    return Shape({1});
}

Shape Shape::sum(int d) const {
    if (d < 0 || d >= _dims.size()) {
        LOG(ERROR) << fmt::format("sum dim out of range: {}, shape: {}", d, _dims.size());
        return Shape();
    }
    std::vector<u32> dims;
    dims.reserve(_dims.size() - 1);
    for (u32 i = 0; i < _dims.size(); i++) {
        if (i == d) {continue;}
        dims.emplace_back(_dims[i]);
    }
    return Shape(dims);
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


Shape Shape::cat(const Shape& t, int d) const {
    if (_dims.size() != t._dims.size()) {
        LOG(ERROR) << fmt::format("cat tensor dim cnt ne: {}, {}", _dims.size(), t._dims.size());
        return Shape();
    }
    if (d < 0 || d >= _dims.size()) {
        LOG(ERROR) << fmt::format("cat d invalid, {}, dim cnt: {}", d, _dims.size());
        return Shape();
    }
    std::vector<u32> dims(_dims.size());
    for (int i = 0; i < _dims.size(); i++) {
        if (i == d) {
            dims[i] = _dims[i] + t._dims[i];
        } else {
            if (_dims[i] != t._dims[i]) {
                LOG(ERROR) << fmt::format("cat dim index: {}, val: {}, {} ne", i, _dims[i], t._dims[i]);
                return Shape();
            }
            dims[i] = _dims[i];
        }
    }
    return Shape(dims);
}


bool Shape::split(int d, u32 sd, Shape& shape1, Shape& shape2) const {
    if (d < 0 || d >= _dims.size()) {
        LOG(ERROR) << fmt::format("split d out of range: {}, dim cnt: {}", d, _dims.size());
        return false;
    }
    if (sd == 0 || sd >= _dims[d]) {
        LOG(ERROR) << fmt::format("split, sd: {}, dim: {}", sd, _dims[d]);
        return false;
    }
    std::vector<u32> dims1(_dims.size()), dims2(_dims.size());
    for (int i = 0; i < _dims.size(); i++) {
        if (i == d) {
            dims1[i] = sd;
            dims2[i] = _dims[i] - sd;
        } else {
            dims1[i] = _dims[i];
            dims2[i] = _dims[i];
        }
    }
    shape1 = Shape(dims1);
    shape2 = Shape(dims2);
    return true;
}

Shape Shape::permute(const std::vector<u32>& pl) const {
    if (pl.size() != _dims.size()) {
        LOG(ERROR) << fmt::format("permute pl len != dim cnt: {}, {}", pl.size(), _dims.size());
        return Shape();
    }
    std::vector<u32> dims(pl.size());
    std::unordered_set<u32> pl_set; pl_set.reserve(pl.size());
    for (u32 i = 0; i < pl.size(); i++) {
        if (pl[i] >= _dims.size()) {
            LOG(ERROR) << fmt::format("permute index out of range: {}, {}", i, pl[i]);
            return Shape();
        }
        if (pl_set.find(pl[i]) != pl_set.end()) {
            LOG(ERROR) << fmt::format("permute index dup: {}, {}", i, pl[i]);
            return Shape();
        }
        pl_set.insert(pl[i]);
        dims[i] = _dims[pl[i]];
    }
    return Shape(dims);
}

Shape Shape::transpose(int d1, int d2) const {
    if (d1 < 0 || d2 < 0 || d1 == d2 || d1 >= _dims.size() || d2 >= _dims.size()) {
        LOG(ERROR) << fmt::format("transpose d1/d2 invalid: {}, {}. shape dim cnt: {}", d1, d2, _dims.size());
        return Shape();
    }
    auto dims = _dims;
    std::swap(dims[d1], dims[d2]);
    return Shape(dims);
}

Shape Shape::mmul(const Shape& s) const {
    if (_dims.size() < 2 || s._dims.size() < 2) {
        LOG(ERROR) << "dim cnt lt 2";
        return Shape();
    }
    if (_dims[_dims.size() - 1] != s._dims[s._dims.size() - 2]) {
        LOG(ERROR) << "can not mul, col cnt != row cnt";
        return Shape();
    }
    std::vector<u32> dims1(_dims.begin(), _dims.end() - 2);
    std::vector<u32> dims2(s._dims.begin(), s._dims.end() - 2);
    if (dims1 != dims2) {
        LOG(ERROR) << "matrix not equal";
        return Shape();
    }

    dims1.push_back(_dims[_dims.size() - 2]);
    dims1.push_back(s._dims[s._dims.size() - 1]);
    return Shape(dims1);
}

}