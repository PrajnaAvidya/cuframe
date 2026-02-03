#pragma once
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/roi_crop.h"
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include <cstdint>

namespace cuframe {

enum class ResizeMode { STRETCH, LETTERBOX };

struct LetterboxInfo {
    float scale_x = 1.0f;   // source pixels per output pixel (horizontal)
    float scale_y = 1.0f;   // source pixels per output pixel (vertical)
    float pad_left = 0.0f;  // letterbox padding in output pixels
    float pad_top = 0.0f;
    float offset_x = 0.0f;  // crop offset in source pixels
    float offset_y = 0.0f;

    // map output coordinates to source frame coordinates
    float to_source_x(float x) const { return (x - pad_left) * scale_x + offset_x; }
    float to_source_y(float y) const { return (y - pad_top) * scale_y + offset_y; }
};

// lightweight descriptor for a retained NV12 frame on device.
// valid from the Pipeline::next() call that produced it until the next next() call.
struct RetainedFrame {
    const uint8_t* data = nullptr;  // NV12 data on device
    int width = 0;
    int height = 0;
    unsigned int pitch = 0;
};

struct PipelineConfig {
    std::string input_path;

    // resize (optional — if not set, output at source resolution)
    bool has_resize = false;
    int resize_width = 0;
    int resize_height = 0;
    ResizeMode resize_mode = ResizeMode::LETTERBOX;
    float pad_value = 114.0f;

    // center crop (applied after resize conceptually, fused in practice)
    bool has_center_crop = false;
    int crop_width = 0;
    int crop_height = 0;

    // normalize (optional — if not set, output in [0,255])
    bool has_normalize = false;
    NormParams norm{};

    // color matrix (auto-selected if not overridden)
    bool auto_color_matrix = true;
    ColorMatrix color_matrix{};

    // channel order
    bool bgr = false;

    // retain decoded NV12 frames for ROI crop
    bool retain_decoded = false;

    // device
    int device_id = 0;

    // batching
    int batch_size = 1;

    // pool
    int pool_size = 2;
};

class Pipeline;

class PipelineBuilder {
public:
    PipelineBuilder& input(const std::string& path);
    PipelineBuilder& resize(int width, int height,
                             ResizeMode mode = ResizeMode::LETTERBOX,
                             float pad_value = 114.0f);
    PipelineBuilder& normalize(std::array<float, 3> mean, std::array<float, 3> std);
    PipelineBuilder& batch(int size);
    PipelineBuilder& pool_size(int n);
    PipelineBuilder& color_matrix(const ColorMatrix& matrix);
    PipelineBuilder& center_crop(int width, int height);
    PipelineBuilder& channel_order_bgr(bool bgr = true);
    PipelineBuilder& retain_decoded(bool retain = true);
    PipelineBuilder& device(int gpu_id);

    Pipeline build();

private:
    PipelineConfig config_;
};

class Pipeline {
public:
    static PipelineBuilder builder();

    // returns next batch, or nullopt at end of stream.
    // the batch data is fully synchronized — ready to use on any stream when returned.
    // batch is refcounted — stays alive as long as any shared_ptr references it.
    // when all refs dropped, batch returns to internal pool.
    std::optional<std::shared_ptr<GpuFrameBatch>> next();

    const PipelineConfig& config() const;

    // source video metadata (delegates to internal demuxer)
    int source_width() const;
    int source_height() const;
    double fps() const;
    int64_t frame_count() const;  // -1 if container doesn't store it

    // transform info for mapping output coordinates back to source pixels
    const LetterboxInfo& letterbox_info() const;

    // internal CUDA stream used for preprocessing.
    // useful for chaining GPU ops (e.g. roi_crop_batch) without a full device sync.
    // valid for the lifetime of the pipeline.
    cudaStream_t stream() const;

    // crop regions from retained NV12 frame, resize + normalize into output batch.
    // requires retain_decoded(true). batch_idx is the frame index in the most recent
    // next() result. uses pipeline's color matrix and stream. output.set_count()
    // called automatically. data is synchronized on return.
    void crop_rois(int batch_idx,
                   const Rect* rois, int num_rois,
                   GpuFrameBatch& output,
                   const NormParams& norm,
                   bool bgr = false);

    void crop_rois(int batch_idx,
                   const std::vector<Rect>& rois,
                   GpuFrameBatch& output,
                   const NormParams& norm,
                   bool bgr = false);

    // retained NV12 frames from the most recent next() call.
    // only populated when retain_decoded(true) was set on the builder.
    // valid until the next next() call. i is batch index [0, retained_count()).
    const RetainedFrame& retained_frame(int i) const;
    int retained_count() const;

    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

private:
    friend class PipelineBuilder;
    Pipeline();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cuframe
