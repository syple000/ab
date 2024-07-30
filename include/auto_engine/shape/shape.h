#ifndef BASE_SHAPE_H 
#define BASE_SHAPE_H

#include "auto_engine/base/basic_types.h"
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

    std::string toString() const;

    u32 tensorSize() const {return _size;}
    u32 subTensorSize(u32 index) const {return _dims.size() > index ? _strides[index] * _dims[index] : 0;}
    const std::vector<u32>& getDims() const {return _dims;}
    u32 getDim(u32 index) const {return index < _dims.size() ? _dims[index] : 0;}
    u32 dimCnt() const {return _dims.size();}
    const std::vector<u32>& getStrides() const {return _strides;}

    Shape sum(u32) const;
    Shape sum() const;
    Shape transpose(u32, u32) const;
    Shape mmul(const Shape&) const;
private:
    void calcStrides();
    std::vector<u32> _dims;
    std::vector<u32> _strides;
    u32 _size = 0;
};
}

#endif
