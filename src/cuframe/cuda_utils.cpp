#include "cuframe/cuda_utils.h"

#include <cuda.h>
#include <stdexcept>
#include <string>

namespace cuframe {

void throw_cuda_error(cudaError_t err, const char* file, int line) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) +
        " cuda error: " + cudaGetErrorString(err));
}

void throw_cu_error(CUresult err, const char* file, int line) {
    const char* name = nullptr;
    const char* str = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &str);
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) +
        " cuda driver error: " + (name ? name : "unknown") +
        " - " + (str ? str : "unknown"));
}

} // namespace cuframe
