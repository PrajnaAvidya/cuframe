#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("device %d: %s (sm_%d%d, %.1f GB)\n",
               i, props.name, props.major, props.minor,
               props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    return 0;
}
