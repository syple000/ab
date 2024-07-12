#ifndef BASE_SLICE_H
#define BASE_SLICE_H

#include "auto_engine/base/exit_code.h"
#include "basic_types.h"
#include "glog/logging.h"
#include <cstdlib>
#include <memory>
#include <vector>

namespace base {

template<typename T>
class Slice {
public:
    Slice() {}
    Slice(std::vector<T>* data): _ref_data(data), _begin_index(0), _len(data->size()) {}
    Slice(std::vector<T>* data, u32 bindex, u32 len): _ref_data(data), _begin_index(bindex), _len(len) {
        if (len + bindex > data->size()) {
            LOG(ERROR) << "slice index out of vec range";
            exit(SLICE_INDEX_INVALID);
        }
    }
    Slice(std::shared_ptr<std::vector<T>> data): _own_data(data), _begin_index(0), _len(data->size()) {}
    Slice(std::shared_ptr<std::vector<T>> data, u32 bindex, u32 len): _own_data(data), _begin_index(bindex), _len(len) {
        if (len + bindex > data->size()) {
            LOG(ERROR) << "slice index out of vec range";
            exit(SLICE_INDEX_INVALID);
        }
    }
    Slice(const Slice<T>&) = default;
    Slice<T>& operator=(const Slice<T>&) = default;
    
    bool operator==(const Slice<T>& s) const {
        if (this == &s) {
            return true;
        }
        if (_len != s._len) {
            return false;
        }
        for (int i = 0; i < _len; i++) {
            if (std::abs(this->operator[](i) - s[i]) > EPSILON) {
                return false;
            }
        }
        return true;
    }

    const T& operator[](u32 index) const { // 先判断区间再调用
        if (index >= _begin_index + _len) {
            LOG(ERROR) << "index out of range";
            exit(SLICE_INDEX_INVALID);
        }
        index = _begin_index + index;
        return getVec()->operator[](index);
    }

    u32 size() const {
        return _len;
    }

    T* data() const {
        auto vec = getVec();
        if (!vec) {return nullptr;}
        auto d = vec->data();
        return d + _begin_index;
    }

private:
    std::vector<T>* _ref_data = nullptr; // 引用，不允许delete
    std::shared_ptr<std::vector<T>> _own_data;
    u32 _begin_index = 0, _len = 0; 

    std::vector<T>* getVec() const {
        if (_ref_data) {
            return _ref_data;
        } else if (_own_data) {
            return _own_data.get();
        } else {
            return nullptr;
        }
    }
};

}

#endif