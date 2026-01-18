#include "cuframe/pipeline.h"
#include "cuframe/batch_pool.h"
#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

extern "C" {
#include <libavcodec/packet.h>
}

#include <stdexcept>
#include <vector>

namespace cuframe {

// ============================================================================
// Pipeline::Impl
// ============================================================================

struct Pipeline::Impl {
    PipelineConfig config;

    // decode
    std::unique_ptr<Demuxer> demuxer;
    std::unique_ptr<Decoder> decoder;

    // preprocess config (resolved at build time)
    ResizeParams resize_params{};
    bool use_fused = false;
    int out_w = 0, out_h = 0;

    // stream
    cudaStream_t preprocess_stream = nullptr;

    // intermediate GPU buffers (allocated once at build, reused per batch)
    std::vector<float*> rgb_bufs;   // only for separate path with resize
    std::vector<float*> out_bufs;   // preprocessed output, one per batch slot

    // output
    std::unique_ptr<BatchPool> batch_pool;

    // decode state — MUST be after decoder for destruction order.
    // PooledBuffers in DecodedFrame reference decoder's FramePool,
    // so pending must be destroyed before decoder.
    std::vector<DecodedFrame> pending;
    int pending_idx = 0;
    bool flushed = false;
    bool eos = false;

    // reusable packet
    AVPacket* packet = nullptr;

    void preprocess(const DecodedFrame& frame, float* out_buf,
                    float* rgb_buf, cudaStream_t stream) const {
        auto* nv12 = static_cast<const uint8_t*>(frame.buffer->data());
        int w = frame.width, h = frame.height;
        unsigned int pitch = frame.pitch;

        if (use_fused) {
            fused_nv12_to_tensor(nv12, out_buf, w, h, pitch,
                resize_params, config.color_matrix, config.norm,
                config.bgr, stream);
        } else if (config.has_resize || config.has_center_crop) {
            // color convert → resize (or crop)
            nv12_to_rgb_planar(nv12, rgb_buf, w, h, pitch,
                               config.color_matrix, config.bgr, stream);
            resize_bilinear(rgb_buf, out_buf, resize_params, stream);
        } else if (config.has_normalize) {
            // color convert → normalize in-place
            nv12_to_rgb_planar(nv12, out_buf, w, h, pitch,
                               config.color_matrix, config.bgr, stream);
            cuframe::normalize(out_buf, out_buf, w, h, config.norm, stream);
        } else {
            // color convert only
            nv12_to_rgb_planar(nv12, out_buf, w, h, pitch,
                               config.color_matrix, config.bgr, stream);
        }
    }

    // pre-decode frames for the next batch while preprocess stream is busy on GPU
    void prefetch() {
        if (flushed) return;

        int target = config.batch_size;
        int have = static_cast<int>(pending.size()) - pending_idx;

        while (have < target) {
            if (!demuxer->read_packet(packet)) {
                decoder->flush(pending);
                flushed = true;
                break;
            }
            decoder->decode(packet, pending);
            av_packet_unref(packet);
            have = static_cast<int>(pending.size()) - pending_idx;
        }
    }

    ~Impl() {
        // release PooledBuffers before decoder dies
        pending.clear();
        for (auto* p : rgb_bufs) cudaFree(p);
        for (auto* p : out_bufs) cudaFree(p);
        if (preprocess_stream) cudaStreamDestroy(preprocess_stream);
        if (packet) av_packet_free(&packet);
    }
};

// ============================================================================
// Pipeline
// ============================================================================

Pipeline::Pipeline() : impl_(nullptr) {}
Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

const PipelineConfig& Pipeline::config() const { return impl_->config; }

PipelineBuilder Pipeline::builder() { return PipelineBuilder{}; }

std::optional<std::shared_ptr<GpuFrameBatch>> Pipeline::next() {
    auto& s = *impl_;
    if (s.eos) return std::nullopt;

    auto batch = s.batch_pool->acquire();
    int collected = 0;

    while (collected < s.config.batch_size) {
        // consume pending decoded frames
        while (s.pending_idx < static_cast<int>(s.pending.size())
               && collected < s.config.batch_size) {
            float* rgb = s.rgb_bufs.empty() ? nullptr : s.rgb_bufs[collected];
            s.preprocess(s.pending[s.pending_idx],
                         s.out_bufs[collected], rgb, s.preprocess_stream);
            s.pending[s.pending_idx] = {};  // release PooledBuffer to decoder pool
            s.pending_idx++;
            collected++;
        }

        // compact when all consumed
        if (s.pending_idx == static_cast<int>(s.pending.size())) {
            s.pending.clear();
            s.pending_idx = 0;
        }

        if (collected >= s.config.batch_size) break;
        if (s.flushed) break;

        // read packets and decode until we get at least one frame
        bool got_frames = false;
        while (s.demuxer->read_packet(s.packet)) {
            int before = static_cast<int>(s.pending.size());
            s.decoder->decode(s.packet, s.pending);
            av_packet_unref(s.packet);
            if (static_cast<int>(s.pending.size()) > before) {
                got_frames = true;
                break;
            }
        }

        if (!got_frames) {
            // demuxer exhausted — flush decoder
            s.decoder->flush(s.pending);
            s.flushed = true;
        }
    }

    if (collected == 0) {
        s.eos = true;
        return std::nullopt;
    }

    // copy preprocessed frames into contiguous batch
    std::vector<const float*> ptrs(collected);
    for (int i = 0; i < collected; ++i)
        ptrs[i] = s.out_bufs[i];
    batch_frames(*batch, ptrs.data(), collected, s.preprocess_stream);

    // pre-decode next batch while preprocess stream finishes on GPU
    s.prefetch();

    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(s.preprocess_stream));
    batch->set_count(collected);
    return batch;
}

// ============================================================================
// PipelineBuilder
// ============================================================================

PipelineBuilder& PipelineBuilder::input(const std::string& path) {
    config_.input_path = path;
    return *this;
}

PipelineBuilder& PipelineBuilder::resize(int width, int height,
                                          ResizeMode mode, float pad_value) {
    config_.has_resize = true;
    config_.resize_width = width;
    config_.resize_height = height;
    config_.resize_mode = mode;
    config_.pad_value = pad_value;
    return *this;
}

PipelineBuilder& PipelineBuilder::normalize(std::array<float, 3> mean,
                                             std::array<float, 3> std) {
    config_.has_normalize = true;
    config_.norm = make_norm_params(mean.data(), std.data());
    return *this;
}

PipelineBuilder& PipelineBuilder::batch(int size) {
    config_.batch_size = size;
    return *this;
}

PipelineBuilder& PipelineBuilder::pool_size(int n) {
    config_.pool_size = n;
    return *this;
}

PipelineBuilder& PipelineBuilder::color_matrix(const ColorMatrix& matrix) {
    config_.auto_color_matrix = false;
    config_.color_matrix = matrix;
    return *this;
}

PipelineBuilder& PipelineBuilder::center_crop(int width, int height) {
    config_.has_center_crop = true;
    config_.crop_width = width;
    config_.crop_height = height;
    return *this;
}

PipelineBuilder& PipelineBuilder::channel_order_bgr(bool bgr) {
    config_.bgr = bgr;
    return *this;
}

Pipeline PipelineBuilder::build() {
    if (config_.input_path.empty())
        throw std::invalid_argument("pipeline: input path required");
    if (config_.batch_size < 1)
        throw std::invalid_argument("pipeline: batch size must be >= 1");

    auto demuxer = std::make_unique<Demuxer>(config_.input_path);
    auto& info = demuxer->video_info();

    // auto color matrix: BT.601 for ≤720p, BT.709 for >720p
    if (config_.auto_color_matrix)
        config_.color_matrix = (info.height > 720) ? BT709 : BT601;

    // validate center crop
    if (config_.has_center_crop) {
        int eff_w = config_.has_resize ? config_.resize_width : info.width;
        int eff_h = config_.has_resize ? config_.resize_height : info.height;
        if (config_.crop_width > eff_w || config_.crop_height > eff_h)
            throw std::invalid_argument(
                "pipeline: crop dimensions exceed resized dimensions");
    }

    int out_w, out_h;
    ResizeParams resize_params{};

    if (config_.has_center_crop) {
        out_w = config_.crop_width;
        out_h = config_.crop_height;
        resize_params = make_center_crop_params(
            info.width, info.height,
            config_.has_resize ? config_.resize_width : 0,
            config_.has_resize ? config_.resize_height : 0,
            config_.crop_width, config_.crop_height);
    } else if (config_.has_resize) {
        out_w = config_.resize_width;
        out_h = config_.resize_height;
        resize_params = (config_.resize_mode == ResizeMode::LETTERBOX)
            ? make_letterbox_params(info.width, info.height,
                  config_.resize_width, config_.resize_height, config_.pad_value)
            : make_resize_params(info.width, info.height,
                  config_.resize_width, config_.resize_height);
    } else {
        out_w = info.width;
        out_h = info.height;
    }

    bool use_fused = (config_.has_resize || config_.has_center_crop)
                     && config_.has_normalize;

    // double-buffer headroom: prefetch decodes batch N+1 while batch N preprocesses
    int pool_count = 2 * config_.batch_size + 4;
    auto decoder = std::make_unique<Decoder>(info, pool_count);

    cudaStream_t preprocess_stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&preprocess_stream));

    // per-frame output buffers (preprocessed result before batching)
    size_t out_bytes = 3ULL * out_w * out_h * sizeof(float);
    std::vector<float*> out_bufs(config_.batch_size, nullptr);
    for (int i = 0; i < config_.batch_size; ++i)
        CUFRAME_CUDA_CHECK(cudaMalloc(&out_bufs[i], out_bytes));

    // intermediate RGB buffers only for separate path with resize/crop
    std::vector<float*> rgb_bufs;
    if ((config_.has_resize || config_.has_center_crop) && !use_fused) {
        size_t rgb_bytes = 3ULL * info.width * info.height * sizeof(float);
        rgb_bufs.resize(config_.batch_size, nullptr);
        for (int i = 0; i < config_.batch_size; ++i)
            CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_bufs[i], rgb_bytes));
    }

    auto batch_pool = std::make_unique<BatchPool>(
        config_.pool_size, config_.batch_size, 3, out_h, out_w);

    // assemble impl
    auto impl = std::make_unique<Pipeline::Impl>();
    impl->config = std::move(config_);
    impl->demuxer = std::move(demuxer);
    impl->decoder = std::move(decoder);
    impl->resize_params = resize_params;
    impl->use_fused = use_fused;
    impl->out_w = out_w;
    impl->out_h = out_h;
    impl->preprocess_stream = preprocess_stream;
    impl->rgb_bufs = std::move(rgb_bufs);
    impl->out_bufs = std::move(out_bufs);
    impl->batch_pool = std::move(batch_pool);
    impl->pending.reserve(config_.batch_size);
    impl->packet = av_packet_alloc();

    Pipeline p;
    p.impl_ = std::move(impl);
    return p;
}

} // namespace cuframe
