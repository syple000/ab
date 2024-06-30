#include "shape.h"
#include "exit_code.h"
#include "glog/logging.h"
#include <cstdlib>
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

}