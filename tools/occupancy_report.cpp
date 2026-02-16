#include <cstdio>
#include <cuda_runtime.h>
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/fused_preprocess.h"

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("cuframe kernel occupancy report\n");
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("max threads/SM: %d  max threads/block: %d  regs/SM: %d\n",
           prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock,
           prop.regsPerMultiprocessor);
    printf("max blocks/SM: %d  shared mem/SM: %zu KB\n\n",
           prop.maxBlocksPerMultiProcessor, prop.sharedMemPerMultiprocessor / 1024);

    struct KernelInfo {
        const char* name;
        void (*query)(int*, int*);
    };

    KernelInfo kernels[] = {
        {"fused_nv12_to_tensor",  cuframe::fused_preprocess_query_occupancy},
        {"nv12_to_rgb_planar",    cuframe::nv12_to_rgb_query_occupancy},
        {"resize_bilinear",       cuframe::resize_query_occupancy},
        {"normalize",             cuframe::normalize_query_occupancy},
    };

    printf("%-30s | %13s | %13s | %s\n",
           "kernel", "optimal_block", "current_block", "change?");
    printf("-------------------------------+---------------+---------------+--------\n");

    for (auto& k : kernels) {
        int min_grid = 0, block_size = 0;
        k.query(&min_grid, &block_size);
        const char* change = (block_size == 256) ? "no" : "YES";
        printf("%-30s | %13d | %13d | %s\n",
               k.name, block_size, 256, change);
    }

    return 0;
}
