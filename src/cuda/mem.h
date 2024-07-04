#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#include "base/basic_types.h"
// #include "folly/AtomicHashMap.h"
// #include "folly/ProducerConsumerQueue.h"

namespace cuda {

#define MAX_CUDA_CACHE_MEM_SIZE 1024*1024

class Mem {
public:
    static void* malloc(u32 size);
    static void free(void* m);
    static void clearAll();
private:
    // static folly::AtomicHashMap<u32, folly::ProducerConsumerQueue<void*>*> _free_mem;
    // static folly::AtomicHashMap<void*, u32> _alloc_mem;

    static void* mallocFromSys(u32 size);
};

}

#endif