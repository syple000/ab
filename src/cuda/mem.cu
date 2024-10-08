#include "auto_engine/cuda/mem.h"
#include "cuda_runtime_api.h"
#include "glog/logging.h"
#include "auto_engine/utils/maths.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <utility>

namespace cuda {

MemConcurrentQueue::MemConcurrentQueue(u32 cap): cb(cap), mutex(std::make_shared<std::mutex>()) {}

bool MemConcurrentQueue::push(void* m) {
    auto guard = std::lock_guard(*mutex);
    if (cb.full()) {
        return false;
    }
    cb.push_back(m);
    return true;
}

void* MemConcurrentQueue::pop() {
    auto guard = std::lock_guard(*mutex);
    if (cb.empty()) {
        return nullptr;
    }
    auto m = cb.back();
    cb.pop_back();
    return m;
}

void MemConcurrentQueue::clearAll() {
    auto guard = std::lock_guard(*mutex);
    while (!cb.empty()) {
        auto m = cb.back();
        cudaFree(m);
        cb.pop_back();
    }
}

MemConcurrentMap::MemConcurrentMap(): mutex(std::make_shared<std::mutex>()){}

bool MemConcurrentMap::insert(void* mem, u32 size) {
    auto guard = std::lock_guard(*mutex);
    m.insert({mem, size});
    return true;
}

u32 MemConcurrentMap::erase(void* mem) {
    auto guard = std::lock_guard(*mutex);
    auto iter = m.find(mem);
    if (iter == m.end()) {
        return 0;
    }
    m.erase(iter);
    return iter->second;
}

#define INIT_FREE_MEM(size) {(size), MemConcurrentQueue(MAX_CUDA_CACHE_MEM_SIZE/(size))}
std::unordered_map<u32, MemConcurrentQueue> Mem::_free_mems = std::unordered_map<u32, MemConcurrentQueue>{
    INIT_FREE_MEM(1<<1), INIT_FREE_MEM(1<<2), INIT_FREE_MEM(1<<3), INIT_FREE_MEM(1<<4),
    INIT_FREE_MEM(1<<5), INIT_FREE_MEM(1<<6), INIT_FREE_MEM(1<<7), INIT_FREE_MEM(1<<8),
    INIT_FREE_MEM(1<<9), INIT_FREE_MEM(1<<10), INIT_FREE_MEM(1<<11), INIT_FREE_MEM(1<<12),
    INIT_FREE_MEM(1<<13), INIT_FREE_MEM(1<<14), INIT_FREE_MEM(1<<15), INIT_FREE_MEM(1<<16),
    INIT_FREE_MEM(1<<17), INIT_FREE_MEM(1<<18), INIT_FREE_MEM(1<<19), INIT_FREE_MEM(1<<20)
};
#undef INIT_FREE_MEM
MemConcurrentMap Mem::_alloc_mems = MemConcurrentMap();

bool Mem::malloc(void** dst, u32 size) {
    if (size == 0) {
        LOG(ERROR) << "malloc zero byte.";
        return false;
    }
    size = utils::nextPowerOfTwo(size);

    auto l = _free_mems.find(size);
    if (l != _free_mems.end()) {
        auto m = l->second.pop();
        if (m) {
            _alloc_mems.insert(m, size);
            *dst = m;
            return true;
        }
    }

    auto m = mallocFromSys(size);
    if (!m) {
        LOG(ERROR) << "malloc size: " << size << " fail";
        return false;
    }

    if (size <= MAX_CUDA_CACHE_MEM_SIZE) {
        _alloc_mems.insert(m, size);
    }
    *dst = m;
    return true;
}

void Mem::free(void* m) {
    auto size = _alloc_mems.erase(m);
    if (size == 0) {
        cudaFree(m);
        return;
    }
    auto iter = _free_mems.find(size);
    if (iter == _free_mems.end()) {
        LOG(ERROR) << "alloc size can not reuse due to size not valid: " << size;
        cudaFree(m);
        return;
    }
    if (!iter->second.push(m)) {
        LOG(INFO) << "alloc size can not reuse due to queue full: " << size;
        cudaFree(m);
        return;
    }
}

void Mem::clearAll() { // 仅清空free列表
    for (auto iter : _free_mems) {
        iter.second.clearAll();
    }
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