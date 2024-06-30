#ifndef BASE_SHAPE_H 
#define BASE_SHAPE_H

#include "basic_types.h"
#include <string>
#include <vector>

namespace base {
class Shape {
public:
    Shape() {};
    Shape(const std::vector<u32>& dims);
    Shape(const Shape&);
    Shape(Shape&&);
    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape);

    bool operator==(const Shape&) const;
    Shape reshape(const std::vector<u32>& dims) const;
    bool reshapeInplace(const std::vector<u32>& dims);

    std::string toString() const;

    u32 tensorSize() const {return _dims.size() > 0 ? _counts[0] : 0;}
    u32 subTensorSize(u32 index) const {return _dims.size() > index ? _counts[index] : 0;}
    const std::vector<u32>& getDims() {return _dims;}
    u32 getDim(u32 index) const {return index < _dims.size() ? _dims[index] : 0;}
    u32 dimCnt() const {return _dims.size();}
private:
    std::vector<u32> _dims;
    std::vector<u32> _counts;
};
}

#endif
