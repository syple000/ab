#include "auto_engine/cuda/mem.h"
#include "cuda_runtime_api.h"
#include "glog/logging.h"
#include "auto_engine/utils/maths.h"
#include <cstdlib>
#include <cuda_runtime.h>

namespace cuda {

// folly::AtomicHashMap<u32, folly::ProducerConsumerQueue<void*>*> Mem::_free_mem(std::log2(MAX_CUDA_CACHE_MEM_SIZE));
// folly::AtomicHashMap<void*, u32> Mem::_alloc_mem(MAX_CUDA_CACHE_MEM_SIZE);

void* Mem::malloc(u32 size) {
    if (size == 0) {
        LOG(ERROR) << "malloc zero byte.";
        return nullptr;
    }
    size = utils::nextPowerOfTwo(size);

//    auto l = _free_mem.find(size);
//    if (l != _free_mem.end()) {
//        void* m;
//        if (l->second->read(m)) {
//            _alloc_mem.insert(m, size);
//            return m; 
//        }
//    }

    auto m = mallocFromSys(size);
    if (!m) {
        LOG(ERROR) << "malloc size: " << size << " fail";
        return nullptr;
    }

    if (size <= MAX_CUDA_CACHE_MEM_SIZE) {
//        _alloc_mem.insert(m, size);
    }
    return m;
}

void Mem::free(void* m) {
//    auto iter = _alloc_mem.find(m);
//    if (iter == _alloc_mem.end()) {
//        cudaFree(m);
//    }
//    auto size = iter->second;
//    _alloc_mem.erase(m);
//
//    if (_free_mem.find(size) == _free_mem.end()) {
//        // 小的块设置更大的buffer，大的块设置少的buffer
//        _free_mem.insert(size, new folly::ProducerConsumerQueue<void*>(MAX_CUDA_CACHE_MEM_SIZE/size));
//    }
//    auto q = _free_mem.find(size)->second;
//    if (!q->write(m)) {
//        cudaFree(m);
//    }
}

void Mem::clearAll() {
//    for (auto m : _alloc_mem) {
//        cudaFree(m.first);
//    }
//    _alloc_mem.clear();
//    for (auto iter : _free_mem) {
//        while(!iter.second->isEmpty()) {
//            void* m;
//            if (iter.second->read(m)) {
//                cudaFree(m);
//            }
//        }
//        delete iter.second;
//    }
//    _free_mem.clear();
}

void* Mem::mallocFromSys(u32 size) {
    void *mem;
    auto err = cudaMalloc(&mem, size);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cuda malloc mem err: " << err;
        return nullptr;
    }
    return mem;
} 

}