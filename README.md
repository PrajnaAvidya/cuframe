# cuframe

GPU video preprocessing library. Decodes video and produces inference-ready tensors entirely on GPU with no unnecessary host copies.

```
[Encoded Video]
      │
      ▼
 ┌──────────┐
 │  NVDEC   │  hardware decode → GPU memory
 └────┬─────┘
      │  NV12 surfaces
      ▼
 ┌──────────────────────────────────┐
 │  Fused Kernel                    │
 │  color convert + resize +        │
 │  normalize (single pass)         │
 └────┬─────────────────────────────┘
      │
      ▼
 ┌──────────┐
 │  Batch   │  N frames → contiguous NCHW tensor
 └────┬─────┘
      │
      ▼
 [float* device pointer]
      │
      ▼
 TensorRT / ONNX Runtime / custom inference
```

## quickstart

```cpp
#include "cuframe/pipeline.h"

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
    .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
    .batch(8)
    .build();

while (auto batch = pipeline.next()) {
    // (*batch)->data() is a float* to NCHW tensor on GPU
    // pass directly to TensorRT, ONNX Runtime, etc.
}
```

## features

- **NVDEC hardware decode** — H.264/HEVC decoded by dedicated GPU hardware, frames land directly in GPU memory
- **fused preprocessing kernel** — NV12 color conversion + bilinear resize + normalize in a single kernel pass, eliminating intermediate buffers
- **auto color matrix** — BT.601 for SD (≤720p), BT.709 for HD (>720p), overridable
- **letterbox resize** — aspect-ratio-preserving resize with configurable pad value (default 114.0 for YOLO convention)
- **refcounted batch pool** — pre-allocated GPU batch buffers returned via `shared_ptr` with custom deleter, supports multiple consumers and backpressure
- **multi-stream prefetch** — overlaps decode of batch N+1 with preprocessing of batch N using separate CUDA streams
- **zero framework dependency** — output is a raw device pointer with shape metadata, works with any CUDA-aware inference framework

## benchmark

1920x1080 H.264 → 640x640 NCHW float32, batch size 8, ImageNet normalization:

| path | fps | vs CPU |
|------|-----|--------|
| cuframe pipeline (multi-stream) | 684 | 9.8x |
| cuframe raw fused kernels | 730 | 10.4x |
| CPU baseline (ffmpeg + sws_scale + cudaMemcpy) | 70 | — |

debug build, 300 frames synthetic content, RTX 3080. fused kernel eliminates 2 intermediate buffers and ~2.5x memory traffic vs separate kernels. pipeline overhead vs raw: ~7% (BatchPool + shared_ptr + state machine).

## build

### requirements

- NVIDIA GPU (tested on Ampere/RTX 3080, sm_86)
- CUDA Toolkit 12.x
- NVIDIA driver with `libnvcuvid.so` (ships with standard driver installs)
- FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`)
- CMake 3.24+
- C++17 compiler

### building

```bash
# configure (must use preset for glibc 2.42+ / CUDA 12.9 compatibility)
cmake --preset default

# build
cmake --build build

# test
ctest --test-dir build

# run examples
./build/examples/basic_pipeline tests/data/test_h264.mp4
```

**note:** the cmake preset is required on systems with glibc 2.42+ (Fedora 43+). it passes `-U_GNU_SOURCE` to nvcc to work around a `noexcept` conflict between glibc's C23 math declarations and CUDA 12.9's `crt/math_functions.h`. see `CMakePresets.json` for details.

### generating a test video

```bash
ffmpeg -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libx264 -pix_fmt yuv420p tests/data/test_h264.mp4
```

## project structure

```
cuframe/
├── CMakeLists.txt
├── CMakePresets.json
├── include/
│   └── cuframe/
│       ├── pipeline.h             # Pipeline, PipelineBuilder (primary API)
│       ├── batch_pool.h           # BatchPool (refcounted batch allocation)
│       ├── gpu_frame_batch.h      # GpuFrameBatch (NCHW tensor)
│       ├── decoder.h              # NVDEC hardware decode wrapper
│       ├── demuxer.h              # FFmpeg container parser
│       ├── device_buffer.h        # RAII GPU memory
│       ├── frame_pool.h           # pre-allocated NV12 buffer pool
│       ├── cuda_utils.h           # error checking macros
│       └── kernels/
│           ├── color_convert.h    # NV12 → RGB planar
│           ├── resize.h           # bilinear resize + letterbox
│           ├── normalize.h        # per-channel normalize
│           └── fused_preprocess.h # single-pass NV12 → tensor
├── src/
│   └── cuframe/
│       ├── pipeline.cpp
│       ├── batch_pool.cpp
│       ├── gpu_frame_batch.cpp
│       ├── decoder.cpp
│       ├── demuxer.cpp
│       ├── device_buffer.cu
│       ├── frame_pool.cpp
│       ├── cuda_utils.cpp
│       └── kernels/
│           ├── color_convert.cu
│           ├── resize.cu
│           ├── normalize.cu
│           └── fused_preprocess.cu
├── tests/
├── benchmarks/
├── examples/
│   ├── basic_pipeline.cpp         # working pipeline example
│   ├── tensorrt_inference.cpp     # TensorRT integration (pseudocode)
│   └── onnxruntime_inference.cpp  # ONNX Runtime integration (pseudocode)
├── docs/
│   └── api.md                     # C++ API reference
└── third_party/
    └── nv-codec-headers/          # vendored NVIDIA Video Codec SDK headers
```

## API reference

see [docs/api.md](docs/api.md) for the full C++ API reference.

## license

MIT — see [LICENSE](LICENSE).
