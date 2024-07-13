#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#include "auto_engine/base/basic_types.h"
#include "boost/circular_buffer.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cuda {

#define MAX_CUDA_CACHE_MEM_SIZE 1024*1024

struct MemConcurrentQueue {
    boost::circular_buffer<void*> cb; 
    std::shared_ptr<std::mutex> mutex;

    MemConcurrentQueue(u32);
    bool push(void* m);
    void* pop();
    void clearAll();
};

struct MemConcurrentMap {
    std::unordered_map<void*, u32> m;
    std::shared_ptr<std::mutex> mutex;

    MemConcurrentMap();
    bool insert(void*, u32);
    u32 erase(void*);   
};

class Mem {
public:
    static void* malloc(u32 size);
    static void free(void* m);
    static void clearAll();

    // host-device数据内存转换
    static void* host2Device(const void* m, u32 size);
    static void* device2Host(const void* m, u32 size);
private:
    static std::unordered_map<u32, MemConcurrentQueue> _free_mems;
    static MemConcurrentMap _alloc_mems;

    static void* mallocFromSys(u32 size);
};

}

#endif