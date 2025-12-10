#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

namespace cuframe {

// throw std::runtime_error with file:line and cuda error string.
// defined in cuda_utils.cpp (not .cu) to avoid nvcc + glibc c23 conflicts.
[[noreturn]] void throw_cuda_error(cudaError_t err, const char* file, int line);
[[noreturn]] void throw_cu_error(CUresult err, const char* file, int line);

} // namespace cuframe

// check cuda runtime api calls, throw on failure
#define CUFRAME_CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) \
        cuframe::throw_cuda_error(err_, __FILE__, __LINE__); \
} while(0)

// check cuda driver api calls, throw on failure
#define CUFRAME_CU_CHECK(call) do { \
    CUresult err_ = (call); \
    if (err_ != CUDA_SUCCESS) \
        cuframe::throw_cu_error(err_, __FILE__, __LINE__); \
} while(0)
