# cuframe

GPU video preprocessing library. Decodes video and produces inference-ready tensors entirely on GPU — one builder, one `next()` call, one device pointer. No intermediate copies, no framework dependencies, no stitching three libraries together.

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

## why

most GPU video preprocessing pipelines require stitching together separate libraries for decode, color conversion, resize, and normalization. each launching its own kernel with its own intermediate buffers. cuframe does all of this in a single fused CUDA kernel pass directly from NVDEC's NV12 output. no existing tool does this, including NVIDIA's DALI and CV-CUDA. preprocessing cost drops to ~0.12ms per frame, and the pipeline runs at ~95% of the NVDEC hardware ceiling.

## quickstart

### python

```bash
pip install .  # from source
```

```python
import cuframe
import torch

pipeline = (cuframe.Pipeline.builder()
    .input("video.mp4")
    .resize(640, 640, cuframe.ResizeMode.LETTERBOX)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .batch(8)
    .build())

for batch in pipeline:
    tensor = torch.from_dlpack(batch)  # zero-copy CUDA tensor
    # tensor: (N, 3, 640, 640), float32, cuda:0
```

`GpuFrameBatch` implements the DLPack protocol — also works directly with ONNX Runtime `io_binding`, CuPy, and JAX. see [examples/python/](examples/python/) for PyTorch, ORT, and two-stage pipeline examples.

### C++

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

### seek + temporal stride

```cpp
// video understanding: 16 frames sampled every 4th frame
auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(224, 224)
    .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
    .batch(16)
    .temporal_stride(4)
    .build();

while (auto clip = pipeline.next()) {
    // each batch spans 16*4 = 64 decoded frames
}

// random access
pipeline.seek(5.0);  // jump to 5 seconds
auto batch = pipeline.next();
```

```python
# same in python
pipeline = (cuframe.Pipeline.builder()
    .input("video.mp4")
    .resize(224, 224)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .batch(16)
    .temporal_stride(4)
    .build())

for clip in pipeline:
    tensor = torch.from_dlpack(clip)

pipeline.seek(5.0)
```

## two-stage pipeline (detect + classify)

for pipelines that need to crop regions from the original frame after first-stage inference (e.g., detect objects then classify each one), use `retain_decoded(true)` to keep the raw NV12 frames, then `roi_crop_batch` to extract and preprocess all detections in a single kernel launch.

```cpp
#include "cuframe/pipeline.h"
#include "cuframe/kernels/roi_crop.h"
#include "cuframe/batch_pool.h"

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
    .normalize({0, 0, 0}, {1, 1, 1})       // YOLO: pixel/255
    .retain_decoded(true)                    // keep NV12 for ROI cropping
    .batch(1)
    .build();

cuframe::BatchPool crop_pool(1, 64, 3, 224, 224);  // up to 64 crops

while (auto batch = pipeline.next()) {
    auto boxes = run_detector((*batch)->data());

    auto& frame = pipeline.retained_frame(0);
    auto crops = crop_pool.acquire();

    std::vector<cuframe::Rect> rois;
    for (auto& b : boxes)
        rois.push_back({b.x, b.y, b.w, b.h});

    // crop + resize + color convert + normalize all ROIs in one kernel
    cuframe::roi_crop_batch(
        frame.data, frame.width, frame.height, frame.pitch,
        rois.data(), rois.size(),
        crops->data(), 224, 224,
        pipeline.config().color_matrix,
        cuframe::make_norm_params({0,0,0}, {1,1,1}),
        false);
    crops->set_count(rois.size());

    auto labels = run_classifier(crops->data(), rois.size());
}
```

the color matrix is read from `pipeline.config().color_matrix` so the ROI crops use the same BT.601/BT.709 auto-selection as the main pipeline. see [docs/api.md](docs/api.md) for the full `roi_crop_batch` API reference.

**ONNX model export note:** for batched ROI inference, the second-stage model needs a dynamic batch dimension. with Ultralytics YOLO, export with `dynamic=True`:

```bash
yolo export model=yolo26n-cls.pt format=onnx imgsz=224 dynamic=True
```

this makes batch, height, and width all symbolic in the ONNX graph. the spatial dimensions can be read from the model's `imgsz` metadata at runtime; the batch dimension accepts any size. without `dynamic=True`, you'll need to run inference per-crop (one `session.Run()` per detection).

## features

- **NVDEC hardware decode** — H.264, H.265/HEVC, VP9, and AV1 decoded by dedicated GPU hardware, frames land directly in GPU memory
- **fused preprocessing kernel** — NV12 color conversion + bilinear resize + normalize in a single kernel pass. no intermediate buffers, no extra kernel launches.
- **auto color matrix** — BT.601 for SD (≤720p), BT.709 for HD (>720p), overridable
- **RGB/BGR channel order** — default RGB, switchable to BGR for OpenCV-convention models
- **center crop** — resize + center crop fused into a single kernel pass for classification pipelines (e.g. resize to 256, crop to 224)
- **letterbox resize** — aspect-ratio-preserving resize with configurable pad value (default 114.0 for YOLO convention)
- **multi-GPU device selection** — pin a pipeline to a specific GPU with `.device(gpu_id)`, one pipeline per GPU
- **refcounted batch pool** — pre-allocated GPU batch buffers returned via `shared_ptr` with custom deleter, supports multiple consumers and backpressure
- **ROI crop** — batch-extract regions from a decoded frame, resize + color convert + normalize each in a single kernel launch. enables two-stage pipelines (detect → crop → classify) without leaving the GPU
- **retained decoded frames** — optionally keep raw NV12 frames alongside preprocessed batches for ROI cropping after first-stage inference. one D2D copy per frame (~0.01ms at 1080p)
- **seek / random access** — jump to any timestamp and resume decoding. precise seek decodes from the nearest keyframe and discards frames before the target. pipelines can be reused after end-of-stream by seeking back
- **temporal stride** — collect every Nth frame for video understanding models (SlowFast, TimeSformer, Video Swin, X3D). `.temporal_stride(4)` gives every 4th frame — skipped frames are released immediately, never preprocessed
- **multi-stream prefetch** — overlaps decode of batch N+1 with preprocessing of batch N using separate CUDA streams
- **NVTX profiling markers** — annotated ranges for decode, preprocess, batch, prefetch, and next. profile with Nsight Systems out of the box. opt-in: no-op if NVTX headers aren't installed
- **zero framework dependency** — output is a raw device pointer with shape metadata, works with any CUDA-aware inference framework

## benchmark

1920x1080 H.264 → 640x640 NCHW float32, batch size 8, ImageNet normalization:

| path | fps | vs CPU |
|------|-----|--------|
| cuframe pipeline (multi-stream) | 684 | 9.8x |
| cuframe raw fused kernels | 730 | 10.4x |
| CPU baseline (ffmpeg + sws_scale + cudaMemcpy) | 70 | — |

debug build, 300 frames synthetic content, RTX 3080 (~717 fps estimated NVDEC ceiling for 1080p H.264). pipeline operates at ~95% of hardware decode limit so preprocessing overhead is effectively zero. fused kernel eliminates 2 intermediate buffers and ~2.5x memory traffic vs separate kernels.

### end-to-end inference

full decode → preprocess → YOLO inference pipeline on real video content. cuframe (NVDEC + fused GPU preprocess, zero-copy to ORT) vs CPU baseline (OpenCV decode + resize/normalize + cudaMemcpy H2D to ORT). both paths run inference on GPU via ONNX Runtime CUDA EP.

test conditions: 1280x696 H.264 live action clip, 24fps, ~1460 frames, RTX 3080. YOLO26n / YOLO11n nano models, batch size 1.

| task | model input | cuframe preprocess | CPU preprocess | preprocess speedup | cuframe e2e fps | CPU e2e fps | e2e speedup |
|------|-------------|-------------------|----------------|-------------------|-----------------|-------------|-------------|
| detect | 640x640 | 0.12ms | 2.75ms | 23x | 227 | 132 | 1.7x |
| pose | 640x640 | 0.12ms | 2.70ms | 23x | 218 | 131 | 1.7x |
| segment | 640x640 | 0.13ms | 3.67ms | 28x | 109 | 75 | 1.5x |
| obb | 640x640 | 0.09ms | 2.68ms | 30x | 265 | 142 | 1.9x |
| classify | 224x224 | 0.12ms | 0.91ms | 8x | 846 | 457 | 1.9x |

preprocessing is 20-30x faster with cuframe. end-to-end speedup is 1.5-1.9x because inference time dominates — but preprocessing drops from ~35% of frame time to <3%, eliminating it as a bottleneck. the CPU path also pays ~0.4ms cudaMemcpy per frame that cuframe avoids via zero-copy.

## build

### requirements

- NVIDIA GPU (Maxwell or newer, sm_50+)
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

the build auto-detects your GPU architecture. to target a specific arch (e.g. for cross-compilation), pass `-DCMAKE_CUDA_ARCHITECTURES=90`.

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
├── pyproject.toml                   # pip install . (scikit-build-core)
├── python/
│   ├── bindings.cpp                 # nanobind C++/Python bindings
│   └── cuframe/
│       └── __init__.py              # Python package
├── include/
│   └── cuframe/
│       ├── cuframe.h                # umbrella header
│       ├── pipeline.h               # Pipeline, PipelineBuilder (primary API)
│       ├── batch_pool.h             # BatchPool (refcounted batch allocation)
│       ├── gpu_frame_batch.h        # GpuFrameBatch (NCHW tensor)
│       ├── decoder.h                # NVDEC hardware decode wrapper
│       ├── demuxer.h                # FFmpeg container parser
│       ├── device_buffer.h          # RAII GPU memory
│       ├── frame_pool.h             # pre-allocated NV12 buffer pool
│       ├── cuda_utils.h             # error checking macros
│       └── kernels/
│           ├── color_convert.h      # NV12 → RGB planar
│           ├── resize.h             # bilinear resize + letterbox
│           ├── normalize.h          # per-channel normalize
│           ├── fused_preprocess.h   # single-pass NV12 → tensor
│           └── roi_crop.h           # batched ROI crop from NV12
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
│           ├── fused_preprocess.cu
│           └── roi_crop.cu
├── tests/
│   └── python/                      # Python binding tests
├── benchmarks/
├── examples/
│   ├── basic_pipeline.cpp           # C++ pipeline example
│   ├── classification_pipeline.cpp  # C++ classification example
│   ├── two_stage_pipeline.cpp       # C++ two-stage example
│   ├── temporal_pipeline.cpp        # C++ temporal stride + seek example
│   ├── tensorrt_inference.cpp       # TensorRT integration (pseudocode)
│   ├── onnxruntime_inference.cpp    # ONNX Runtime integration (pseudocode)
│   └── python/
│       ├── basic_pipeline.py        # Python pipeline + DLPack
│       ├── two_stage_pipeline.py    # detect → crop → classify
│       ├── temporal_pipeline.py     # temporal stride + seek
│       └── ort_inference.py         # ORT io_binding (no torch needed)
├── docs/
│   └── api.md                       # API reference (Python + C++)
└── third_party/
    └── nv-codec-headers/            # vendored NVIDIA Video Codec SDK headers
```

## API reference

see [docs/api.md](docs/api.md) for the full API reference (Python and C++).

## license

MIT — see [LICENSE](LICENSE).
