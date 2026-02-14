# cuframe API reference

## Python API

### installation

```bash
pip install .  # from source (requires CUDA toolkit + FFmpeg dev packages)
```

for development builds without `pip install`:
```bash
cmake --preset default && cmake --build build
PYTHONPATH=python:build/src python -c "import cuframe; print('ok')"
```

### quick start

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
    tensor = torch.from_dlpack(batch)
    # tensor: (N, 3, 640, 640), float32, cuda:0
```

the Python API mirrors the C++ API. see the C++ sections below for full documentation of each builder method and class.

### seek

jump to an arbitrary timestamp and resume decoding from there. useful for video understanding workloads (sparse frame sampling), random access, or replaying a section.

```python
pipeline.seek(1.5)  # jump to 1.5 seconds
batch = pipeline.next()  # frames starting at/after 1.5s

# reuse a pipeline after exhaustion
for batch in pipeline:
    pass  # drain
pipeline.seek(0.0)  # back to start
for batch in pipeline:
    pass  # works again
```

seek always lands on the keyframe at or before the target timestamp. the pipeline then internally decodes forward from the keyframe and discards frames before the target, so the first frame returned by `next()` is at or just past the requested time. this "precise seek" is transparent to the caller.

calling `seek()` resets end-of-stream state — a pipeline that returned `None` becomes usable again.

### temporal stride

sample every Nth frame instead of consecutive frames. enables video understanding models (SlowFast, TimeSformer, Video Swin, X3D) that need temporally spaced frames.

```python
pipeline = (cuframe.Pipeline.builder()
    .input("video.mp4")
    .resize(224, 224)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .batch(16)
    .temporal_stride(4)  # every 4th frame
    .build())

for clip in pipeline:
    tensor = torch.from_dlpack(clip)  # [count, 3, 224, 224]
```

stride=1 (default) gives consecutive frames. stride=2 gives every other frame, etc. stride resets on `seek()` — the first frame after a seek is always collected.

### error recovery

by default, cuframe throws on any decode or demux error. for production deployments processing videos that may be corrupt or truncated, enable skip-and-continue mode:

```python
pipeline = (cuframe.Pipeline.builder()
    .input("maybe_corrupt.mp4")
    .resize(640, 640)
    .normalize(cuframe.YOLO_NORM)
    .batch(8)
    .error_policy(cuframe.ErrorPolicy.SKIP)
    .on_error(lambda info: print(f"skipped: {info.message}"))
    .build())

for batch in pipeline:
    tensor = torch.from_dlpack(batch)
    # process...

print(f"total errors: {pipeline.error_count}")
```

- `ErrorPolicy.THROW` (default) — any demux/decode error raises an exception. backward compatible.
- `ErrorPolicy.SKIP` — corrupt packets and frames are skipped. after a decode error, the NVDEC parser is reset and decoding resumes at the next keyframe (losing up to one GOP, typically 1-5 seconds).
- `on_error(callback)` — optional. called for each skipped error with an `ErrorInfo` object containing a `message` string.
- `error_count` — total number of errors encountered over the pipeline's lifetime. not reset by `seek()`.
- if the error callback itself raises, the exception propagates to the caller. use this for "stop after N errors" policies.
- CUDA device errors and resource exhaustion (pool/memory) remain fatal regardless of error policy.

### DLPack zero-copy

`GpuFrameBatch` implements the DLPack protocol (`__dlpack__`, `__dlpack_device__`). the batch's GPU memory stays alive as long as any downstream tensor references it.

**PyTorch path:**
```python
tensor = torch.from_dlpack(batch)  # zero-copy CUDA tensor
output = model(tensor)
```

**ONNX Runtime path** (no torch dependency needed):
```python
io = session.io_binding()
io.bind_input("images", "cuda", 0, np.float32,
              list(batch.shape), batch.data_ptr)
for name in output_names:
    io.bind_output(name, "cuda")
session.run_with_iobinding(io)
```

also works with CuPy (`cupy.from_dlpack(batch)`), JAX, and any other DLPack consumer.

### iteration and frame_count

always iterate with `for batch in pipeline:` — this works for all container formats.

**do not** use `range(pipeline.frame_count)` — `frame_count` returns -1 for containers that don't store frame counts (MKV, some WebM). the `for batch in pipeline:` pattern handles end-of-stream automatically.

```python
# correct — always works
for batch in pipeline:
    process(batch)

# also correct — explicit end-of-stream check
batch = pipeline.next()  # returns None at end
```

`Pipeline` does not implement `__len__`. use `pipeline.frame_count` if you need the total frame count (with a -1 check):

```python
total = pipeline.frame_count if pipeline.frame_count >= 0 else None
```

### crop_rois — tuple support

`crop_rois` accepts plain tuples `(x, y, w, h)` in addition to `Rect` objects. both can be mixed in the same list:

```python
rois = [(10, 20, 100, 80), cuframe.Rect(50, 30, 60, 60)]
pipeline.crop_rois(0, rois, crops, cuframe.YOLO_NORM)
```

### GpuFrameBatch properties

| property | type | description |
|----------|------|-------------|
| `count` | int | valid frames in this batch (may be < batch_size for the last batch) |
| `batch_size` | int | allocated capacity |
| `channels` | int | always 3 (RGB or BGR) |
| `height` | int | output height in pixels |
| `width` | int | output width in pixels |
| `shape` | tuple | `(count, channels, height, width)` — pass to `list(batch.shape)` for ORT |
| `data_ptr` | int | raw CUDA device pointer — pass to ORT `io_binding.bind_input()` |

### differences from C++ API

- pipeline is iterable: `for batch in pipeline:` (vs C++ `while (auto batch = pipeline.next())`)
- `pipeline.next()` returns `None` at end-of-stream (not `std::nullopt`)
- `pipeline.seek(seconds)` works the same as C++ (method call, not property)
- metadata via properties: `pipeline.source_width` (not `pipeline.source_width()`)
- `pipeline.error_count` is a property (not `pipeline.error_count()`)
- `crop_rois` accepts a list of `Rect` objects or `(x, y, w, h)` tuples
- `make_norm_params(mean, std)` takes lists: `make_norm_params([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

### constants

- `cuframe.IMAGENET_NORM` — ImageNet normalization (mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225})
- `cuframe.YOLO_NORM` — YOLO normalization (pixel / 255.0)

---

## C++ API

## Pipeline API

the pipeline is the primary interface. it wraps decode, preprocessing, and batching into a pull-based iterator.

### PipelineBuilder

```cpp
#include <cuframe/cuframe.h>  // everything, or include individual headers below

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")                                              // required
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX, 114.0f)        // optional
    .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})   // optional
    .batch(8)                                                        // optional (default 1)
    .pool_size(2)                                                    // optional (default 2)
    .color_matrix(cuframe::BT709)                                    // optional (auto-detected)
    .build();
```

**`input(const std::string& path)`** — video file path. required. supports any container FFmpeg can demux (MP4, MKV, WebM, MOV, AVI, etc.) with H.264, H.265/HEVC, VP9, or AV1 video. 8-bit and 10-bit content supported (10-bit decoded from P016 surfaces).

**`resize(int width, int height, ResizeMode mode, float pad_value)`** — optional. resize all frames to the target dimensions.
- `ResizeMode::LETTERBOX` (default) — scale to fit preserving aspect ratio, center the image, pad borders with `pad_value`
- `ResizeMode::STRETCH` — stretch to exact dimensions, ignoring aspect ratio
- `pad_value` defaults to 114.0f (YOLO convention, in [0, 255] range)
- if omitted, output is at source resolution

**`center_crop(int width, int height)`** — optional. center crop the (conceptually) resized image to width × height.
- if resize is also set: resize first, then crop the center region. the two operations are fused into a single kernel pass — no intermediate buffer.
- if resize is not set: crop from the center of the source frame at source resolution.
- crop dimensions must not exceed the resized (or source) dimensions.
- typical classification pipeline: `.resize(256, 256, ResizeMode::STRETCH).center_crop(224, 224).normalize(...)`

**`normalize(std::array<float, 3> mean, std::array<float, 3> std)`** — optional. per-channel normalization. transforms pixel values from [0, 255] to `(pixel/255 - mean) / std`. typical values:
- ImageNet: mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
- if omitted, output is in [0, 255] float32 range

**`batch(int size)`** — optional (default 1). number of frames per batch. the last batch may have fewer frames (check `count()`).

**`pool_size(int n)`** — optional (default 2). number of pre-allocated batch buffers. controls backpressure — if all batches are held by the caller, `next()` blocks until one is released. 2 is enough for double-buffering.

**`color_matrix(const ColorMatrix& matrix)`** — optional. override the NV12 → RGB color matrix. if not set, auto-selects BT.601 for source ≤720p and BT.709 for >720p. use `cuframe::BT601` or `cuframe::BT709`.

**`channel_order_bgr(bool bgr = true)`** — optional. output BGR channel order instead of RGB. use for models trained with OpenCV defaults (Caffe, PaddlePaddle, some YOLO variants). default is RGB.

**`retain_decoded(bool retain = true)`** — optional. when enabled, keeps a copy of the raw NV12 decoded frames alongside each preprocessed batch. use this for two-stage inference pipelines where you need to crop regions from the original frame after running a detector. access retained frames via `Pipeline::retained_frame(i)`. adds one D2D copy per frame (~0.01ms at 1080p) and `batch_size` NV12 buffers (~3.1MB each at 1080p).

**`device(int gpu_id)`** — optional (default 0). select which GPU to run on. all decode, preprocessing, and output buffers are allocated on this device. for multi-GPU setups, create one pipeline per GPU. invalid device IDs throw `std::runtime_error` at build time.

**`temporal_stride(int stride)`** — optional (default 1). collect every Nth decoded frame instead of consecutive frames. stride=1 gives every frame, stride=2 gives every other frame, etc. enables video understanding models that need temporally spaced frames (SlowFast, TimeSformer, etc.). the decoder still decodes every frame (hardware constraint), but skipped frames are released immediately and never preprocessed. throws `std::invalid_argument` if stride < 1.

**`error_policy(ErrorPolicy policy)`** — optional (default `ErrorPolicy::THROW`). controls how demux/decode errors are handled.
- `ErrorPolicy::THROW` — errors throw `std::runtime_error` (current behavior, backward compatible)
- `ErrorPolicy::SKIP` — corrupt packets/frames are skipped, pipeline continues to next keyframe. after a decode error, the NVDEC parser is reset (losing frames in the DPB, typically up to one GOP). the demuxer continues feeding packets, and NVDEC resumes output at the next keyframe. gives up after 100 consecutive errors to prevent infinite loops on fully corrupt files.

**`on_error(ErrorCallback callback)`** — optional. called for each error skipped in SKIP mode. receives an `ErrorInfo` struct with a `message` field. if the callback throws, the exception propagates to the caller — use this to implement "stop after N errors" or custom logging.

**`build()`** — creates the pipeline. opens the video file, initializes the decoder, allocates all GPU buffers. throws `std::invalid_argument` if no input path is set, `std::runtime_error` if the file can't be opened or decoded.

### Pipeline

```cpp
class Pipeline {
public:
    static PipelineBuilder builder();
    std::optional<std::shared_ptr<GpuFrameBatch>> next();
    void seek(double seconds);
    const PipelineConfig& config() const;
    int source_width() const;
    int source_height() const;
    double fps() const;
    int64_t frame_count() const;
    const LetterboxInfo& letterbox_info() const;
    size_t error_count() const;
    cudaStream_t stream() const;
    void crop_rois(int batch_idx, const Rect* rois, int num_rois,
                   GpuFrameBatch& output, const NormParams& norm, bool bgr = false);
    void crop_rois(int batch_idx, const std::vector<Rect>& rois,
                   GpuFrameBatch& output, const NormParams& norm, bool bgr = false);
};
```

**`next()`** — returns the next batch of preprocessed frames, or `std::nullopt` at end of stream. the batch data is fully synchronized when returned — you can use it immediately on any stream, pass it to TensorRT/ONNX Runtime, or memcpy to host without any additional synchronization. you do NOT need to call `cudaDeviceSynchronize()` after `next()`.

the returned `shared_ptr<GpuFrameBatch>` has a custom deleter that returns the batch to an internal pool when all references are dropped. this means:
- you can copy the `shared_ptr` freely (e.g., pass to an async inference engine while preparing the next batch)
- GPU memory is not freed — just marked available for reuse
- if all pool slots are occupied, `next()` blocks until a batch is released (backpressure)

the last batch may be partial: `(*batch)->count()` may be less than `(*batch)->batch_size()`.

**`seek(double seconds)`** — jump to the given timestamp. the next `next()` call returns frames starting at or after this position. seeks to the nearest keyframe at or before the target and discards intermediate frames internally (precise seek). resets end-of-stream state — a pipeline that returned `nullopt` becomes usable again after seeking. also resets temporal stride state so the first frame after a seek is always collected. thread-unsafe — don't call during `next()`.

**`config()`** — returns the resolved pipeline configuration.

**`source_width()`** / **`source_height()`** — source video resolution in pixels (before any resize/crop).

**`fps()`** — average frame rate of the source video. computed from FFmpeg's `avg_frame_rate`.

**`frame_count()`** — total number of frames in the source video, or -1 if the container doesn't store this information.

**`letterbox_info()`** — returns the coordinate transform for mapping output pixel coordinates back to source frame coordinates. useful for reversing letterbox/resize/crop transforms on detection boxes.

**`error_count()`** — total number of errors encountered in SKIP mode. cumulative over the pipeline's lifetime — not reset by `seek()`. returns 0 if no errors occurred or if using THROW mode.

**`stream()`** — returns the pipeline's internal CUDA stream. useful for chaining GPU operations (e.g. running `roi_crop_batch` on the same stream) without a full device sync between pipeline output and downstream kernels. the returned stream is valid for the lifetime of the pipeline.

**`crop_rois(batch_idx, rois, num_rois, output, norm, bgr)`** — crop regions from a retained NV12 frame, resize + normalize into an output batch. convenience wrapper around `roi_crop_batch()` that handles frame lookup, color matrix selection, stream management, and count tracking automatically.

- `batch_idx` — frame index in the most recent `next()` result (0 for batch_size=1)
- `rois` / `num_rois` — bounding boxes in source pixel coordinates (host array)
- `output` — pre-allocated `GpuFrameBatch` (from a `BatchPool`). `dst_w = output.width()`, `dst_h = output.height()`
- `norm` — normalization params (e.g. `IMAGENET_NORM`). can differ from the pipeline's own norm
- `bgr` — output BGR instead of RGB (default false)
- requires `retain_decoded(true)` — throws `std::logic_error` otherwise
- throws `std::out_of_range` if batch_idx is invalid
- runs on the pipeline's internal stream, synchronized on return
- `output.set_count(num_rois)` called automatically

also accepts `const std::vector<Rect>&` instead of pointer + count.

### LetterboxInfo

```cpp
struct LetterboxInfo {
    float scale_x = 1.0f;   // source pixels per output pixel (horizontal)
    float scale_y = 1.0f;   // source pixels per output pixel (vertical)
    float pad_left = 0.0f;  // letterbox padding in output pixels
    float pad_top = 0.0f;
    float offset_x = 0.0f;  // crop offset in source pixels
    float offset_y = 0.0f;

    float to_source_x(float x) const;
    float to_source_y(float y) const;
};
```

general formula: `source = (output - pad) * scale + offset`

- **letterbox**: `pad > 0` on one axis, `offset = 0`. typical detection pipeline — subtract padding, scale to source.
- **stretch**: `pad = 0`, `offset = 0`, `scale_x != scale_y`. just scale.
- **center crop**: `pad = 0`, `offset > 0` (crop origin in source pixels).
- **no resize**: identity — `scale = 1`, `pad = 0`, `offset = 0`.

```cpp
auto& lb = pipeline.letterbox_info();
// map detector output box to source video coordinates
float src_x = lb.to_source_x(det_x);
float src_y = lb.to_source_y(det_y);
float src_w = det_w * lb.scale_x;
float src_h = det_h * lb.scale_y;
```

**`retained_frame(int i)`** — returns a `RetainedFrame` descriptor for the `i`-th frame in the most recent batch. only valid when `retain_decoded(true)` was set. valid until the next `next()` call.

**`retained_count()`** — number of retained frames available (matches the most recent batch's `count()`). returns 0 if `retain_decoded` is not enabled.

### RetainedFrame

```cpp
struct RetainedFrame {
    const uint8_t* data;    // NV12 data on device
    int width, height;
    unsigned int pitch;     // row stride in bytes
};
```

lightweight descriptor for a retained NV12 decoded frame. the `data` pointer is valid from the `next()` call that produced it until the next `next()` call.

### PipelineConfig

```cpp
struct PipelineConfig {
    std::string input_path;
    bool has_resize = false;
    int resize_width = 0;
    int resize_height = 0;
    ResizeMode resize_mode = ResizeMode::LETTERBOX;
    float pad_value = 114.0f;
    bool has_center_crop = false;
    int crop_width = 0;
    int crop_height = 0;
    bool has_normalize = false;
    NormParams norm{};
    bool auto_color_matrix = true;
    ColorMatrix color_matrix{};
    bool bgr = false;
    bool retain_decoded = false;
    int device_id = 0;
    int batch_size = 1;
    int pool_size = 2;
    int temporal_stride = 1;
    ErrorPolicy error_policy = ErrorPolicy::THROW;
    ErrorCallback error_callback;
};
```

### ErrorPolicy

```cpp
enum class ErrorPolicy { THROW, SKIP };
```

### ErrorInfo

```cpp
struct ErrorInfo {
    std::string message;
};

using ErrorCallback = std::function<void(const ErrorInfo&)>;
```

### ResizeMode

```cpp
enum class ResizeMode { STRETCH, LETTERBOX };
```

---

## GpuFrameBatch

```cpp
#include "cuframe/gpu_frame_batch.h"
```

contiguous NCHW float32 tensor in GPU memory. move-only.

```cpp
class GpuFrameBatch {
public:
    GpuFrameBatch(int n, int c, int h, int w);

    float* data() const;          // device pointer to start of tensor
    float* frame(int i) const;    // device pointer to frame i (offset by i*C*H*W)

    int batch_size() const;       // allocated capacity
    int count() const;            // valid frames (may be < batch_size for last batch)
    void set_count(int n);        // set valid frame count

    int channels() const;
    int height() const;
    int width() const;
    size_t frame_size_bytes() const;  // C * H * W * sizeof(float)
    size_t total_size_bytes() const;  // N * C * H * W * sizeof(float)
};
```

**`data()`** — device pointer to the NCHW tensor. pass this to inference frameworks. layout: `batch_size × channels × height × width` contiguous float32.

**`frame(i)`** — device pointer to frame `i` within the batch. equivalent to `data() + i * channels() * height() * width()`.

**`count()` vs `batch_size()`** — `batch_size()` is the allocated capacity (immutable). `count()` is the number of valid frames (mutable, set by the pipeline). for the last batch of a video, `count() < batch_size()`. always use `count()` to determine how many frames to process.

### batch_frames

```cpp
void batch_frames(
    GpuFrameBatch& batch,
    const float* const* frame_ptrs,  // host array of device pointers
    int n,
    cudaStream_t stream = nullptr
);
```

copies `n` float32 planar frames into a contiguous batch via async D2D memcpy. `frame_ptrs` is a host-side array where each element is a device pointer to a `C × H × W` float buffer.

---

## preprocessing kernels

individual kernels for building custom pipelines. the pipeline API uses these internally — most users don't need them directly.

### color conversion

```cpp
#include "cuframe/kernels/color_convert.h"

cuframe::nv12_to_rgb_planar(nv12_ptr, rgb_ptr, width, height, pitch,
                             cuframe::BT601, stream);
```

converts NV12 pitched GPU buffer to float32 RGB planar in [0, 255] range.

- `nv12_ptr` — device pointer to NV12 data (luma plane then chroma plane, pitched layout)
- `rgb_ptr` — output device pointer, must be `3 * width * height * sizeof(float)` bytes
- `pitch` — row stride in bytes of the NV12 buffer (≥ width, typically 256-byte aligned)
- `matrix` — color conversion coefficients. `cuframe::BT601` for SD, `cuframe::BT709` for HD

```cpp
struct ColorMatrix {
    float3 r;  // {Y_coeff, U_coeff, V_coeff} for R channel
    float3 g;  // for G channel
    float3 b;  // for B channel
};

extern const ColorMatrix BT601;  // SD content (≤720p)
extern const ColorMatrix BT709;  // HD content (>720p)
```

### resize

```cpp
#include "cuframe/kernels/resize.h"

auto params = cuframe::make_letterbox_params(src_w, src_h, 640, 640);
cuframe::resize_bilinear(src_ptr, dst_ptr, params, stream);
```

bilinear resize with optional letterbox padding. operates on float32 RGB planar data.

- `make_resize_params(src_w, src_h, dst_w, dst_h)` — plain stretch resize
- `make_letterbox_params(src_w, src_h, dst_w, dst_h, pad_value)` — aspect-ratio-preserving resize with centered padding. `pad_value` defaults to 114.0f.
- pixel center alignment follows the OpenCV `INTER_LINEAR` convention: `(x + 0.5) * scale - 0.5`

```cpp
struct ResizeParams {
    int src_w, src_h;
    int dst_w, dst_h;        // output dimensions (including padding)
    int pad_left, pad_top;   // padding offsets
    int inner_w, inner_h;    // scaled image dimensions
    float scale_x, scale_y;  // source pixels per output pixel
    float pad_value;
    float src_offset_x = 0.0f;  // crop offset in source pixels
    float src_offset_y = 0.0f;
};
```

- `make_center_crop_params(src_w, src_h, resize_w, resize_h, crop_w, crop_h)` — compute resize params for center crop. if `resize_w/resize_h` are 0, crop directly from source. otherwise, maps the center crop region of the (conceptually) resized image back to source coordinates.

### normalize

```cpp
#include "cuframe/kernels/normalize.h"

cuframe::normalize(src_ptr, dst_ptr, width, height,
                    cuframe::IMAGENET_NORM, stream);
```

per-channel scale+bias normalization. transforms [0, 255] → normalized range using a single FMA per pixel: `output = input * scale + bias`.

- supports in-place: `src_ptr == dst_ptr`
- `make_norm_params(mean, std)` pre-computes `scale[c] = 1/(255*std[c])`, `bias[c] = -mean[c]/std[c]`
- `IMAGENET_NORM` — pre-computed for ImageNet mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
- `YOLO_NORM` — pre-computed for YOLO-style: mean={0, 0, 0}, std={1, 1, 1} (just divides by 255)

```cpp
struct NormParams {
    float scale[3];  // per-channel: 1.0 / (255.0 * std)
    float bias[3];   // per-channel: -mean / std
};

extern const NormParams IMAGENET_NORM;
extern const NormParams YOLO_NORM;   // mean={0,0,0}, std={1,1,1} — pixel / 255.0
NormParams make_norm_params(const float mean[3], const float std[3]);
```

### fused preprocessing

```cpp
#include "cuframe/kernels/fused_preprocess.h"

cuframe::fused_nv12_to_tensor(nv12_ptr, rgb_ptr,
    src_w, src_h, src_pitch,
    resize_params, color_matrix, norm_params, stream);
```

NV12 → normalized float32 RGB planar in one kernel launch. color conversion, bilinear resize, and normalization combined. eliminates 2 intermediate buffers and ~2.5x memory traffic vs separate kernels.

the pipeline uses this automatically when both resize and normalize are configured.

### ROI crop

```cpp
#include "cuframe/kernels/roi_crop.h"

cuframe::Rect rois[] = {{10, 10, 100, 80}, {200, 50, 60, 120}};
cuframe::roi_crop_batch(nv12_ptr, src_w, src_h, src_pitch,
    rois, 2, output_ptr, 224, 224,
    cuframe::BT601, cuframe::IMAGENET_NORM, false, stream);
```

batch crop: extract multiple regions from a single NV12 frame, resize each to `dst_w x dst_h`, color convert, and normalize in a single kernel launch. uses a 3D grid with `blockIdx.z` indexing ROIs for minimal launch overhead regardless of crop count.

- `rois` — host-side array of `Rect` structs (uploaded to device internally via stream-ordered alloc)
- output layout: `output[roi * 3 * dst_h * dst_w + c * dst_h * dst_w + y * dst_w + x]` — contiguous NCHW, same as `GpuFrameBatch`
- no padding: ROI crops are always stretched to `dst_w x dst_h`
- source coordinates are clamped to frame bounds (safe for ROIs at edges)
- `num_rois == 0` is a no-op

```cpp
struct Rect {
    int x, y, w, h;  // bounding box in source pixel coordinates
};
```

**two-stage pipeline example** (detect + classify):

```cpp
auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
    .normalize({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})
    .retain_decoded(true)
    .batch(1)
    .build();

cuframe::BatchPool crop_pool(1, 64, 3, 224, 224);
const auto& cfg = pipeline.config();

while (auto batch = pipeline.next()) {
    auto boxes = run_detector((*batch)->data(), 640, 640);

    std::vector<cuframe::Rect> rois;
    for (auto& b : boxes)
        rois.push_back({b.x, b.y, b.w, b.h});

    auto crops = crop_pool.acquire();
    pipeline.crop_rois(0, rois, *crops, cfg.norm);

    auto labels = run_classifier(crops->data(), rois.size());
}
```

**tips:**
- use `pipeline.letterbox_info()` to map detection coordinates from output space back to source pixels before constructing `Rect` values for ROI cropping. the `to_source_x/y` methods handle letterbox padding, resize scaling, and crop offsets.
- use `pipeline.config().color_matrix` so ROI crops use the same auto-selected BT.601/BT.709 as the main pipeline
- use `pipeline.config().norm` to reuse the same normalization, or pass different `NormParams` if the second-stage model expects different normalization (e.g. ImageNet vs YOLO)
- the crop output is contiguous NCHW, same layout as `GpuFrameBatch`. if the second-stage model supports dynamic batch, pass `crops->data()` directly with batch size = `rois.size()`. for ONNX models, export with `dynamic=True` (e.g. `yolo export ... dynamic=True`) to enable this. without dynamic batch, loop over `crops->frame(i)` and run inference per-crop
- for pipelines using `retain_decoded(true)`, prefer `Pipeline::crop_rois()` which handles frame lookup, color matrix, stream, and count automatically. use `roi_crop_batch()` directly when working with non-pipeline NV12 sources or custom streams.

---

## low-level components

for building custom decode/preprocessing pipelines outside of the Pipeline API.

### Demuxer

```cpp
#include "cuframe/demuxer.h"

cuframe::Demuxer demuxer("video.mp4");
auto& info = demuxer.video_info();

AVPacket* pkt = av_packet_alloc();
while (demuxer.read_packet(pkt)) {
    // pkt contains an encoded video packet in annex-b format
    av_packet_unref(pkt);
}
av_packet_free(&pkt);
```

opens a video file via FFmpeg `libavformat`. yields encoded video packets with automatic annex-b bitstream conversion for MP4/MKV/FLV containers. skips non-video streams internally.

```cpp
struct VideoInfo {
    int width, height;
    double fps;                        // average frame rate
    AVCodecID codec_id;
    AVRational time_base;
    int64_t num_frames;                // -1 if unknown
    std::vector<uint8_t> extradata;    // SPS/PPS for H.264, VPS/SPS/PPS for HEVC
};
```

### Decoder

```cpp
#include "cuframe/decoder.h"

cuframe::Decoder decoder(demuxer.video_info(), /* pool_count */ 8);

std::vector<cuframe::DecodedFrame> frames;
decoder.decode(pkt, frames);   // may produce 0+ frames per packet
decoder.flush(frames);         // drain remaining frames at EOF
```

NVDEC hardware decode wrapper. callback-driven architecture (sequence/decode/display). produces `DecodedFrame` structs with NV12 GPU buffers from an internal pool.

**important:** all `DecodedFrame`s must be destroyed before the `Decoder`. the `PooledBuffer` inside each frame auto-returns to the decoder's pool — if the pool is already destroyed, this is undefined behavior. declare frames after the decoder so C++ destruction order handles it.

```cpp
struct DecodedFrame {
    PooledBuffer buffer;        // NV12 data in GPU memory (auto-returns to pool)
    int width, height;
    unsigned int pitch;         // row stride in bytes
    int64_t timestamp;
    CUstream stream;            // decoder's CUDA stream
};
```

### DeviceBuffer

```cpp
#include "cuframe/device_buffer.h"

cuframe::DeviceBuffer buf(1024 * 1024);  // 1MB GPU allocation
void* ptr = buf.data();
size_t sz = buf.size();
// freed automatically on destruction
```

RAII GPU memory wrapper. uses `cudaMallocAsync`/`cudaFreeAsync` when a stream is provided, falls back to sync `cudaMalloc`/`cudaFree` otherwise. move-only, no copy.

### FramePool / PooledBuffer

```cpp
#include "cuframe/frame_pool.h"

cuframe::FramePool pool(frame_size_bytes, 8);
cuframe::PooledBuffer buf = pool.acquire();  // borrows from pool
// buf auto-returns to pool when destroyed
```

pre-allocated pool of fixed-size GPU buffers. `PooledBuffer` is a `unique_ptr<DeviceBuffer, PoolDeleter>` — when it goes out of scope, the buffer returns to the pool rather than being freed.

### BatchPool

```cpp
#include "cuframe/batch_pool.h"

cuframe::BatchPool pool(2, 8, 3, 640, 640);  // 2 slots, batch 8, 3x640x640
auto batch = pool.acquire();      // blocks until a slot is free
auto batch2 = pool.try_acquire(); // non-blocking, returns nullptr if exhausted
// shared_ptr — batch returns to pool when all references are dropped
```

pre-allocated pool of `GpuFrameBatch` objects. `acquire()` returns a `shared_ptr<GpuFrameBatch>` with a custom deleter that marks the slot as free (GPU memory is NOT freed — the pool owns it for its lifetime). supports multiple consumers via `shared_ptr` copy semantics.

---

## error handling

all CUDA API calls are checked via `CUFRAME_CUDA_CHECK` (runtime API) and `CUFRAME_CU_CHECK` (driver API). on failure, these throw `std::runtime_error` with the file, line, and CUDA error string.

```cpp
#include "cuframe/cuda_utils.h"

CUFRAME_CUDA_CHECK(cudaMalloc(&ptr, size));
CUFRAME_CU_CHECK(cuCtxGetCurrent(&ctx));
```

NVDEC callbacks (sequence, decode, display) never throw through NVDEC's C code. exceptions are caught inside callbacks, stored in `std::exception_ptr`, and re-thrown after `cuvidParseVideoData()` returns. this avoids undefined behavior from unwinding C++ exceptions through C stack frames.

frame pool exhaustion throws `std::logic_error` (not `std::runtime_error`), so it is never caught by the pipeline's error recovery. pool exhaustion indicates a configuration issue, not stream corruption.

for production use with potentially corrupt video, see `ErrorPolicy::SKIP` in the Pipeline builder docs above.

---

## NVTX profiling

```cpp
#include "cuframe/nvtx.h"
```

cuframe emits NVTX range markers around pipeline stages, visible in NVIDIA Nsight Systems. annotated ranges:

| range | location | description |
|-------|----------|-------------|
| `cuframe::next` | `Pipeline::next()` | full pipeline iteration (decode + preprocess + batch) |
| `cuframe::decode` | `Decoder::decode()` | NVDEC hardware decode |
| `cuframe::preprocess` | pipeline internals | fused/separate kernel dispatch |
| `cuframe::batch` | `batch_frames()` | D2D copy into contiguous NCHW tensor |
| `cuframe::prefetch` | pipeline internals | async decode of next batch |

**opt-in via header detection.** if `<nvtx3/nvToolsExt.h>` is found at compile time, markers are active. otherwise they compile to no-ops — zero overhead when NVTX isn't installed.

```bash
# profile a cuframe application with nsight systems
nsys profile ./my_app
nsys-ui report.nsys-rep
```

the macros are available for user code as well:

```cpp
#include "cuframe/nvtx.h"

CUFRAME_NVTX_PUSH("my_inference");
// ... run inference ...
CUFRAME_NVTX_POP();
```
