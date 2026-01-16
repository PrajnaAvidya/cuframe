# cuframe C++ API reference

## Pipeline API

the pipeline is the primary interface. it wraps decode, preprocessing, and batching into a pull-based iterator.

### PipelineBuilder

```cpp
#include "cuframe/pipeline.h"

auto pipeline = cuframe::Pipeline::builder()
    .input("video.mp4")                                              // required
    .resize(640, 640, cuframe::ResizeMode::LETTERBOX, 114.0f)        // optional
    .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})   // optional
    .batch(8)                                                        // optional (default 1)
    .pool_size(2)                                                    // optional (default 2)
    .color_matrix(cuframe::BT709)                                    // optional (auto-detected)
    .build();
```

**`input(const std::string& path)`** — video file path. required. supports any container FFmpeg can demux (MP4, MKV, AVI, etc.) with H.264 or HEVC video.

**`resize(int width, int height, ResizeMode mode, float pad_value)`** — optional. resize all frames to the target dimensions.
- `ResizeMode::LETTERBOX` (default) — scale to fit preserving aspect ratio, center the image, pad borders with `pad_value`
- `ResizeMode::STRETCH` — stretch to exact dimensions, ignoring aspect ratio
- `pad_value` defaults to 114.0f (YOLO convention, in [0, 255] range)
- if omitted, output is at source resolution

**`normalize(std::array<float, 3> mean, std::array<float, 3> std)`** — optional. per-channel normalization. transforms pixel values from [0, 255] to `(pixel/255 - mean) / std`. typical values:
- ImageNet: mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
- if omitted, output is in [0, 255] float32 range

**`batch(int size)`** — optional (default 1). number of frames per batch. the last batch may have fewer frames (check `count()`).

**`pool_size(int n)`** — optional (default 2). number of pre-allocated batch buffers. controls backpressure — if all batches are held by the caller, `next()` blocks until one is released. 2 is enough for double-buffering.

**`color_matrix(const ColorMatrix& matrix)`** — optional. override the NV12 → RGB color matrix. if not set, auto-selects BT.601 for source ≤720p and BT.709 for >720p. use `cuframe::BT601` or `cuframe::BT709`.

**`build()`** — creates the pipeline. opens the video file, initializes the decoder, allocates all GPU buffers. throws `std::invalid_argument` if no input path is set, `std::runtime_error` if the file can't be opened or decoded.

### Pipeline

```cpp
class Pipeline {
public:
    static PipelineBuilder builder();
    std::optional<std::shared_ptr<GpuFrameBatch>> next();
    const PipelineConfig& config() const;
};
```

**`next()`** — returns the next batch of preprocessed frames, or `std::nullopt` at end of stream.

the returned `shared_ptr<GpuFrameBatch>` has a custom deleter that returns the batch to an internal pool when all references are dropped. this means:
- you can copy the `shared_ptr` freely (e.g., pass to an async inference engine while preparing the next batch)
- GPU memory is not freed — just marked available for reuse
- if all pool slots are occupied, `next()` blocks until a batch is released (backpressure)

the last batch may be partial: `(*batch)->count()` may be less than `(*batch)->batch_size()`.

**`config()`** — returns the resolved pipeline configuration.

### PipelineConfig

```cpp
struct PipelineConfig {
    std::string input_path;
    bool has_resize = false;
    int resize_width = 0;
    int resize_height = 0;
    ResizeMode resize_mode = ResizeMode::LETTERBOX;
    float pad_value = 114.0f;
    bool has_normalize = false;
    NormParams norm{};
    bool auto_color_matrix = true;
    ColorMatrix color_matrix{};
    int batch_size = 1;
    int pool_size = 2;
};
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

**`data()`** — device pointer to the beginning of the NCHW tensor. pass this to inference frameworks. layout: `batch_size × channels × height × width` contiguous float32.

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
};
```

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

```cpp
struct NormParams {
    float scale[3];  // per-channel: 1.0 / (255.0 * std)
    float bias[3];   // per-channel: -mean / std
};

extern const NormParams IMAGENET_NORM;
NormParams make_norm_params(const float mean[3], const float std[3]);
```

### fused preprocessing

```cpp
#include "cuframe/kernels/fused_preprocess.h"

cuframe::fused_nv12_to_tensor(nv12_ptr, rgb_ptr,
    src_w, src_h, src_pitch,
    resize_params, color_matrix, norm_params, stream);
```

single-pass NV12 → normalized float32 RGB planar. combines color conversion, bilinear resize, and normalization in one kernel launch. eliminates 2 intermediate buffers and ~2.5x memory traffic vs separate kernels.

the pipeline uses this automatically when both resize and normalize are configured.

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
