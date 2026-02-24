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

most GPU video preprocessing pipelines require stitching together separate libraries for decode, color conversion, resize, and normalization. each launching its own kernel with its own intermediate buffers. cuframe does all of this in a single fused CUDA kernel pass directly from NVDEC's NV12 output. no existing tool does this, including NVIDIA's DALI and CV-CUDA. preprocessing adds <1% overhead vs raw NVDEC decode.

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
    .normalize(cuframe.IMAGENET_NORM)
    .batch(8)
    .build())

for batch in pipeline:
    tensor = torch.from_dlpack(batch)  # zero-copy CUDA tensor
    # tensor: (N, 3, 640, 640), float32, cuda:0
```

`GpuFrameBatch` implements the DLPack protocol — also works directly with ONNX Runtime `io_binding`, CuPy, and JAX. see [examples/python/](examples/python/) for PyTorch, ORT, and two-stage pipeline examples.

### C++

```cpp
#include <cuframe/cuframe.h>

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
    .normalize(cuframe::IMAGENET_NORM)
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
    .normalize(cuframe::IMAGENET_NORM)
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
    .normalize(cuframe.IMAGENET_NORM)
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
#include <cuframe/cuframe.h>

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
    .normalize(cuframe::YOLO_NORM)           // YOLO: pixel/255
    .retain_decoded(true)                    // keep NV12 for ROI cropping
    .batch(1)
    .build();

cuframe::BatchPool crop_pool(1, 64, 3, 224, 224);  // up to 64 crops
auto& lb = pipeline.letterbox_info();

while (auto batch = pipeline.next()) {
    auto boxes = run_detector((*batch)->data());

    // map detector output coords back to source pixels (undo letterbox)
    std::vector<cuframe::Rect> rois;
    for (auto& b : boxes)
        rois.push_back({(int)lb.to_source_x(b.x), (int)lb.to_source_y(b.y),
                        (int)(b.w * lb.scale_x), (int)(b.h * lb.scale_y)});

    // crop + resize + color convert + normalize all ROIs in one kernel launch
    auto crops = crop_pool.acquire();
    pipeline.crop_rois(0, rois, *crops, cuframe::IMAGENET_NORM);

    auto labels = run_classifier(crops->data(), rois.size());
}
```

```python
# Python — tuples instead of Rect objects
lb = pipeline.letterbox_info
crops = crop_pool.acquire()
pipeline.crop_rois(0,
    [(int(lb.to_source_x(b.x)), int(lb.to_source_y(b.y)),
      int(b.w * lb.scale_x), int(b.h * lb.scale_y)) for b in boxes],
    crops, cuframe.IMAGENET_NORM)
labels = run_classifier(crops.data_ptr, len(boxes))
```

`pipeline.crop_rois()` handles frame lookup, color matrix, stream management, and count tracking automatically. `pipeline.letterbox_info()` maps detection coordinates from the letterboxed output space back to source pixels. see [docs/api.md](docs/api.md) and [examples/python/two_stage_pipeline.py](examples/python/two_stage_pipeline.py) for full examples.

**ONNX model export note:** for batched ROI inference, the second-stage model needs a dynamic batch dimension. with Ultralytics YOLO, export with `dynamic=True`:

```bash
yolo export model=yolo26n-cls.pt format=onnx imgsz=224 dynamic=True
```

this makes batch, height, and width all symbolic in the ONNX graph. the spatial dimensions can be read from the model's `imgsz` metadata at runtime; the batch dimension accepts any size. without `dynamic=True`, you'll need to run inference per-crop (one `session.Run()` per detection).

## error recovery

for production workloads processing videos that may be corrupt or truncated, enable skip-and-continue mode:

```cpp
auto pipeline = cuframe::Pipeline::builder()
    .input("maybe_corrupt.mp4")
    .resize(640, 640)
    .normalize(cuframe::IMAGENET_NORM)
    .batch(8)
    .error_policy(cuframe::ErrorPolicy::SKIP)
    .on_error([](const cuframe::ErrorInfo& e) {
        fprintf(stderr, "skipped: %s\n", e.message.c_str());
    })
    .build();

while (auto batch = pipeline.next()) {
    // (*batch)->data() — process normally, corrupt frames already skipped
}
printf("total errors: %zu\n", pipeline.error_count());
```

```python
pipeline = (cuframe.Pipeline.builder()
    .input("maybe_corrupt.mp4")
    .resize(640, 640)
    .normalize(cuframe.IMAGENET_NORM)
    .batch(8)
    .error_policy(cuframe.ErrorPolicy.SKIP)
    .on_error(lambda e: print(f"skipped: {e.message}"))
    .build())

for batch in pipeline:
    tensor = torch.from_dlpack(batch)

print(f"total errors: {pipeline.error_count}")
```

`ErrorPolicy::THROW` (default) raises on any decode error. `ErrorPolicy::SKIP` silently skips corrupt packets and resets the decoder to the next keyframe — typically losing up to one GOP. CUDA device errors and resource exhaustion remain fatal regardless of policy.

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
- **error recovery** — `ErrorPolicy::SKIP` with optional `on_error` callback for corrupt streams. skips bad packets, resets decoder to next keyframe. CUDA errors remain fatal. `error_count()` tracks cumulative skips
- **multi-stream prefetch** — overlaps decode of batch N+1 with preprocessing of batch N using separate CUDA streams
- **event-based sync** — `Pipeline::next()` uses CUDA events for batch-ready signaling. `batch_event()` enables GPU→GPU cross-stream sync without CPU round-trips (e.g. inference stream waits on preprocessing completion via `cudaStreamWaitEvent`)
- **profiling-driven kernel tuning** — all kernels annotated with `__launch_bounds__` and `__restrict__` based on ncu occupancy analysis. `tools/occupancy_report` queries `cudaOccupancyMaxPotentialBlockSize` for validation
- **NVTX profiling markers** — annotated ranges for decode, preprocess, batch, prefetch, and next. profile with Nsight Systems out of the box. opt-in: no-op if NVTX headers aren't installed
- **zero framework dependency** — output is a raw device pointer with shape metadata, works with any CUDA-aware inference framework

## benchmark

1920x1080 H.264 → 640x640 NCHW float32, batch size 8, ImageNet normalization. RTX 3080, 16k frames real video content (11 min H.264 TV episode):

| path | fps |
|------|-----|
| cuframe pipeline | 768 |
| raw fused kernels (no pipeline) | 766 |
| decode only (no preprocessing) | 757 |
| [NVIDIA published NVDEC spec](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html) (Ampere, 1080p H.264) | 748 |

preprocessing adds <1% overhead vs decode-only throughput. the pipeline is NVDEC-bottlenecked and fused kernel preprocessing is effectively free. pipeline slightly exceeds decode-only due to multi-stream prefetch overlap (CPU feeds packets while GPU finishes the current batch). exceeding the NVIDIA spec is expected as their published number is a conservative baseline across content types.

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

## accuracy

cuframe's GPU preprocessing pipeline (NVDEC → NV12→RGB fused kernel → letterbox → normalize) is compared against an OpenCV CPU reference (software decode → BGR → cvtColor → resize → /255) using the same ONNX model and video.

### preprocessed tensor comparison

preprocessed tensor (NCHW float32, [0,1]) produced by cuframe GPU vs OpenCV CPU, per-pixel absolute error across 200 frames:

| video | codec | resolution | color matrix | mean error | max error | PSNR |
|-------|-------|------------|--------------|------------|-----------|------|
| live action | H.264 | 1280×696 | BT.601 | 0.018 (4.6/255) | 0.071 (18.2/255) | 31.3 dB |
| live action | HEVC 8-bit | 1280×696 | BT.601 | 0.017 (4.4/255) | 0.073 (18.5/255) | 31.4 dB |
| live action | VP9 | 1280×696 | BT.601 | 0.017 (4.4/255) | 0.073 (18.5/255) | 31.4 dB |
| live action | HEVC 10-bit | 1280×696 | BT.601 | 0.015 (3.9/255) | 0.080 (20.5/255) | 32.3 dB |
| synthetic | H.264 | 1920×1080 | BT.709 | 0.037 (9.4/255) | 0.207 (52.8/255) | 23.3 dB |

source of delta: NVDEC hardware decode and OpenCV software decode use different YCbCr→RGB color matrices. cuframe auto-selects BT.601 for ≤720 lines and BT.709 for >720 lines; OpenCV's decoder applies its own fixed matrix. for real-world content, this produces ~4–5/255 mean error. the synthetic 1080p row is a worst case, fully saturated test pattern colors are maximally sensitive to matrix differences.

everything else matches exactly: same bilinear interpolation coefficients (pixel-center aligned, matching `cv::INTER_LINEAR`), same letterbox geometry (`min(dst_w/src_w, dst_h/src_h)` with pad=114), same normalization (pixel/255).

### model output comparison

same video, same model, same inputs as above (YOLO26n-pose). detections filtered at conf > 0.25, boxes matched greedily by IoU:

| video | codec | GPU dets | CPU dets | matched | mean IoU | score MAE |
|-------|-------|----------|----------|---------|----------|-----------|
| live action | H.264 | 401 | 401 | 400 | 0.989 | 0.017 |
| live action | HEVC 8-bit | 411 | 412 | 409 | 0.990 | 0.009 |
| live action | VP9 | 415 | 410 | 409 | 0.990 | 0.006 |
| live action | HEVC 10-bit | 413 | 413 | 407 | 0.989 | 0.013 |

classification (YOLO26n-cls, 200 frames, center-crop 224×224):

| video | codec | top-1 agreement | top-5 agreement | score MAE |
|-------|-------|-----------------|-----------------|-----------|
| live action | H.264 | 200/200 (100%) | 200/200 (100%) | 0.000058 |
| live action | HEVC 10-bit | 189/200 (94%) | 188/200 (94%) | 0.000145 |

for real-world video, cuframe GPU preprocessing produces model outputs effectively identical to an OpenCV CPU reference. box IoU > 0.989, classification top-1 agreement ≥ 94%, score differences in the third decimal place.

## design decisions

brief notes on key architectural choices. see the [API reference](docs/api.md) for full documentation.

**fused kernel** — NV12 color conversion, bilinear resize, and normalization are combined into a single CUDA kernel pass. the pipeline auto-selects the fused path when both resize (or center crop) and normalize are configured. this eliminates two intermediate GPU buffers and ~2.5x memory traffic vs separate kernels. the fallback (separate kernels) is used for color-convert-only or normalize-only configurations.

**stored exception pattern** — NVDEC invokes C++ callbacks from C code. throwing C++ exceptions through C stack frames is undefined behavior. cuframe catches exceptions inside callbacks, stores them via `std::exception_ptr`, and re-throws after `cuvidParseVideoData()` returns to C++ context.

**deferred unmap** — after the async D2D copy from NVDEC surfaces to our pool buffers, the NVDEC surface is not immediately unmapped. instead, a CUDA event is recorded and the unmap is deferred until the event signals completion. this allows NVDEC hardware to start decoding the next frame while the copy is still in flight.

**event-based sync** — `Pipeline::next()` uses `cudaEventRecord` + `cudaEventSynchronize` instead of `cudaStreamSynchronize`. this is more composable — downstream inference streams can `cudaStreamWaitEvent(their_stream, pipeline.batch_event())` for GPU-to-GPU synchronization without blocking the CPU.

**DLPack lifetime management** — the Python bindings use a `DLPackContext` that holds both a `shared_ptr<GpuFrameBatch>` (prevents pool return) and a raw `PyObject*` reference to the pipeline (prevents pool destruction). the DLPack deleter manually acquires the GIL via `PyGILState_Ensure` because PyTorch may call the deleter from a non-Python thread.

**bilinear interpolation in RGB space** — the fused kernel converts each of the 4 NV12 neighbor pixels to RGB, then blends. the strictly correct approach is to interpolate in YUV space first, then convert once. the difference is negligible for inference preprocessing (measured <0.1% mAP impact) and the RGB approach avoids an extra conversion step.

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
│   ├── error_recovery.cpp           # C++ error recovery with SKIP policy
│   ├── tensorrt_inference.cpp       # TensorRT integration (pseudocode)
│   ├── onnxruntime_inference.cpp    # ONNX Runtime integration (pseudocode)
│   └── python/
│       ├── basic_pipeline.py        # Python pipeline + DLPack
│       ├── two_stage_pipeline.py    # detect → crop → classify
│       ├── temporal_pipeline.py     # temporal stride + seek
│       ├── error_recovery.py        # error recovery with SKIP policy
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
