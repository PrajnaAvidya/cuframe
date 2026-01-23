#pragma once
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include <string>
#include <array>
#include <memory>
#include <optional>

namespace cuframe {

enum class ResizeMode { STRETCH, LETTERBOX };

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
    PipelineBuilder& device(int gpu_id);

    Pipeline build();

private:
    PipelineConfig config_;
};

class Pipeline {
public:
    static PipelineBuilder builder();

    // returns next batch, or nullopt at end of stream.
    // batch is refcounted — stays alive as long as any shared_ptr references it.
    // when all refs dropped, batch returns to internal pool.
    std::optional<std::shared_ptr<GpuFrameBatch>> next();

    const PipelineConfig& config() const;

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
