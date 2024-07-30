#include "auto_engine/tensor/shape.h"
#include "auto_engine/base/exit_code.h"
#include "glog/logging.h"
#include <cstdlib>
#include <fmt/core.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace base {

std::vector<u32> calc_counts(const std::vector<u32> dims) {
    if (dims.size() == 0) {
        return {};
    }
    std::vector<u32> counts(dims.size());
    counts[dims.size()-1] = dims[dims.size()-1];
    for (int i = dims.size() - 2; i >= 0; i--) {
        counts[i] = counts[i+1] * dims[i];
        if (counts[i] < counts[i+1] || counts[i] < dims[i]) {
            LOG(ERROR) << __FUNCTION__ << "shape size overflow";
            exit(SHAPE_SIZE_OVERFLOW);
        }
    }
    return counts;
}

Shape::Shape(const std::vector<u32>& dims) {
    auto counts = calc_counts(dims);
    _dims = dims;
    _counts = std::move(counts);
}

Shape::Shape(const Shape& shape) {
    _dims = shape._dims;
    _counts = shape._counts;
}

Shape::Shape(Shape&& shape) {
    _dims = std::move(shape._dims);
    _counts = std::move(shape._counts);
}

Shape& Shape::operator=(const Shape& shape) {
    if (this == &shape) {
        return *this;
    } 
    _dims = shape._dims;
    _counts = shape._counts;
    return *this;
}

Shape& Shape::operator=(Shape&& shape) {
    if (this == &shape) {
        return *this;
    } 
    _dims = std::move(shape._dims);
    _counts = std::move(shape._counts);
    return *this;
}

bool Shape::operator==(const Shape& shape) const {
    return _dims == shape._dims;
}

bool check_reshapeable(const std::vector<u32>& dims1, const std::vector<u32>& dims2) {
    auto counts1 = calc_counts(dims1);
    auto counts2 = calc_counts(dims2);
    if (!counts1.size() && !counts2.size()) {
        return true;
    }
    if (!counts1.size() || !counts2.size()) {
        return false;
    }
    return counts1[0] == counts2[0];
}

Shape Shape::reshape(const std::vector<u32>& dims) const {
    if (!check_reshapeable(_dims, dims)) {
        LOG(ERROR) << "check reshapeable fail";
        return Shape();
    }
    return Shape(dims);
}

bool Shape::reshapeInplace(const std::vector<u32>& dims) {
    if (!check_reshapeable(_dims, dims)) {
        LOG(ERROR) << "check reshapeable fail";
        return false;
    }
    auto counts = calc_counts(dims);
    _dims = dims;
    _counts = std::move(counts);
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


Shape Shape::sumAlongRow() const {
    if (_dims.size() < 2) {
        LOG(ERROR) << "sumAlongRow non-matrix";
        return Shape();
    }
    auto dims = std::vector<u32>(_dims.begin(), _dims.end() - 1);
    return Shape(dims);
}

Shape Shape::sumAlongCol() const {
    if (_dims.size() < 2) {
        LOG(ERROR) << "sumAlongCol non-matrix";
        return Shape();
    }
    auto dims = std::vector<u32>(_dims.begin(), _dims.end() - 2);
    dims.push_back(_dims[_dims.size() - 1]);
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