#ifndef BASE_SLICE_H
#define BASE_SLICE_H

#include "auto_engine/base/exit_code.h"
#include "basic_types.h"
#include "glog/logging.h"
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <vector>

namespace base {

template<typename T>
class Slice {
public:
    Slice() {}
    Slice(std::initializer_list<T> l): _own_data(std::make_shared<std::vector<T>>(l)), _begin_index(0), _end_index(l.size()) {}
    Slice(u32 size): _own_data(std::make_shared<std::vector<T>>(size)), _begin_index(0), _end_index(size) {}
    Slice(u32 size, const T& t): _own_data(std::make_shared<std::vector<T>>(size, t)), _begin_index(0), _end_index(size) {}
    Slice(std::shared_ptr<std::vector<T>> data): _own_data(data), _begin_index(0), _end_index(data->size()) {}
    Slice(std::shared_ptr<std::vector<T>> data, u32 bindex, u32 eindex): _own_data(data), _begin_index(bindex), _end_index(eindex) {
        if (eindex > data->size()) {
            LOG(ERROR) << "slice index out of vec range";
            exit(SLICE_INDEX_INVALID);
        }
        if (bindex > eindex) {
            LOG(ERROR) << "slice start > end";
            exit(SLICE_INDEX_INVALID);
        }
    }
    Slice(const Slice<T>&) = default;
    Slice(Slice<T>&&) = default;
    Slice<T>& operator=(const Slice<T>&) = default;
    Slice<T>& operator=(Slice<T>&&) = default;
    
    bool operator==(const Slice<T>& s) const {
        if (this == &s) {
            return true;
        }
        if (_end_index - _begin_index != s._end_index - s._begin_index) {
            return false;
        }
        for (int i = 0; i < _end_index - _begin_index; i++) {
            if (std::abs(_own_data->operator[](i+_begin_index) - s._own_data->operator[](i+s._begin_index)) > EPSILON) {
                return false;
            }
        }
        return true;
    }

    Slice<T> sub(u32 start, u32 end) const { // [start, end)在slice的基础上截取
        if (start > end) {
            LOG(ERROR) << "sub slice start > end";
            exit(SLICE_INDEX_INVALID);
        }
        if (end + _begin_index > _end_index) {
            LOG(ERROR) << "sub slice end out of range";
            exit(SLICE_INDEX_INVALID);
        }
        return Slice<T>(_own_data, _begin_index + start, _begin_index + end);
    }

    T& operator[](u32 index) {
        if (index >= _end_index - _begin_index) {
            LOG(ERROR) << "index out of range";
            exit(SLICE_INDEX_INVALID);
        }
        index = _begin_index + index;
        return _own_data->operator[](index);
    }

    const T& operator[](u32 index) const {
        if (index >= _end_index - _begin_index) {
            LOG(ERROR) << "index out of range";
            exit(SLICE_INDEX_INVALID);
        }
        index = _begin_index + index;
        return _own_data->operator[](index);
    }

    u32 size() const {
        return _end_index - _begin_index;
    }

    const T* data() const {
        return _own_data->data() + _begin_index;
    }
    T* data() {
        return _own_data->data() + _begin_index;
    }
private:
    std::shared_ptr<std::vector<T>> _own_data;
    u32 _begin_index = 0, _end_index = 0; 
};

}

#endif