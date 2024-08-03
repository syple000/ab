#ifndef BASE_SHAPE_H 
#define BASE_SHAPE_H

#include "auto_engine/base/basic_types.h"
#include <functional>
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

    std::string toString() const;

    u32 tensorSize() const {return _size;}
    u32 subTensorSize(u32 index) const {return _dims.size() > index ? _strides[index] * _dims[index] : 0;}
    const std::vector<u32>& getDims() const {return _dims;}
    u32 getDim(u32 index) const {return index < _dims.size() ? _dims[index] : 0;}
    u32 dimCnt() const {return _dims.size();}
    const std::vector<u32>& getStrides() const {return _strides;}

    bool reshape(const std::vector<u32>& dims, Shape& shape) const;
    bool sum(int, Shape& shape) const;
    bool expand(const Shape&, int) const; // 仅检查
    bool sum(Shape&) const;
    bool expand(const Shape&) const; // 仅检查
    static bool cat(const std::vector<std::reference_wrapper<Shape>>&, int d, Shape&);
    static bool split(const Shape&, const std::vector<u32>&, u32 d, std::vector<Shape>&);
    bool cat(const Shape&, int d, u32 d_offset);
    bool split(const Shape&, u32 d, u32 d_offset);
    bool permute(const std::vector<u32>&, Shape&) const;
    bool transpose(int, int, Shape&) const;
    bool mmul(const Shape&, Shape&) const;
private:
    void calcStrides();
    std::vector<u32> _dims;
    std::vector<u32> _strides;
    u32 _size = 0;
};
}

#endif
