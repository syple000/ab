#include "auto_engine/shape/shape.h"
#include "auto_engine/base/exit_code.h"
#include "glog/logging.h"
#include <cstdlib>
#include <fmt/core.h>
#include <sstream>
#include <string>
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

Shape Shape::sum(u32 d) const {
    if (d >= _dims.size()) {
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

Shape Shape::transpose(u32 d1, u32 d2) const {
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