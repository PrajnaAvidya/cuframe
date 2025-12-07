#include "cuframe/cuda_utils.h"

#include <stdexcept>
#include <string>

namespace cuframe {

void throw_cuda_error(cudaError_t err, const char* file, int line) {
    throw std::runtime_error(
        std::string(file) + ":" + std::to_string(line) +
        " cuda error: " + cudaGetErrorString(err));
}

} // namespace cuframe
