#include "cuframe/pipeline.h"
#include "cuframe/nvtx.h"
#include "cuframe/batch_pool.h"
#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/roi_crop.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

extern "C" {
#include <libavcodec/packet.h>
}

#include <algorithm>
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
    LetterboxInfo letterbox;
    bool use_fused = false;
    int out_w = 0, out_h = 0;

    // stream + sync
    cudaStream_t preprocess_stream = nullptr;
    cudaEvent_t batch_ready = nullptr;

    // intermediate GPU buffers (allocated once at build, reused per batch)
    std::vector<float*> rgb_bufs;   // only for separate path with resize
    std::vector<float*> out_bufs;   // preprocessed output, one per batch slot
    std::vector<const float*> batch_ptrs;  // reused in next() for batch_frames()

    // retained NV12 (only when config.retain_decoded)
    bool retained_allocated = false;
    std::vector<uint8_t*> retained_nv12;
    std::vector<RetainedFrame> retained_meta;
    size_t retained_frame_bytes = 0;
    int retained_count_ = 0;

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

    // seek state (precise seek — discard pre-target frames in next())
    bool seeking = false;
    int64_t seek_target_pts = 0;

    // temporal stride state
    int stride_skip = 0;  // frames to skip before collecting next

    // error recovery
    size_t error_count = 0;

    void preprocess(const DecodedFrame& frame, float* out_buf,
                    float* rgb_buf, cudaStream_t stream) const {
        CUFRAME_NVTX_PUSH("cuframe::preprocess");
        auto* nv12 = static_cast<const uint8_t*>(frame.buffer->data());
        int w = frame.width, h = frame.height;
        unsigned int pitch = frame.pitch;

        bool is_10bit = frame.bit_depth > 8;

        if (use_fused) {
            fused_nv12_to_tensor(nv12, out_buf, w, h, pitch,
                resize_params, config.color_matrix, config.norm,
                config.bgr, is_10bit, stream);
        } else if (config.has_resize || config.has_center_crop) {
            // color convert → resize (or crop)
            nv12_to_rgb_planar(nv12, rgb_buf, w, h, pitch,
                               config.color_matrix, config.bgr, is_10bit, stream);
            resize_bilinear(rgb_buf, out_buf, resize_params, stream);
        } else if (config.has_normalize) {
            // color convert → normalize in-place
            nv12_to_rgb_planar(nv12, out_buf, w, h, pitch,
                               config.color_matrix, config.bgr, is_10bit, stream);
            cuframe::normalize(out_buf, out_buf, w, h, config.norm, stream);
        } else {
            // color convert only
            nv12_to_rgb_planar(nv12, out_buf, w, h, pitch,
                               config.color_matrix, config.bgr, is_10bit, stream);
        }
        CUFRAME_NVTX_POP();
    }

    // pre-decode frames for the next batch while preprocess stream is busy on GPU
    void prefetch() {
        if (flushed) return;
        CUFRAME_NVTX_PUSH("cuframe::prefetch");

        // with temporal stride, need stride*batch_size decoded frames to fill a batch.
        // cap at 4x batch_size to avoid huge pending vectors with large strides.
        int target = std::min(config.batch_size * config.temporal_stride,
                              config.batch_size * 4);
        int have = static_cast<int>(pending.size()) - pending_idx;
        int consecutive_errors = 0;

        while (have < target) {
            bool have_packet;
            try {
                have_packet = demuxer->read_packet(packet);
                consecutive_errors = 0;
            } catch (const std::runtime_error& e) {
                if (config.error_policy == ErrorPolicy::THROW) throw;
                error_count++;
                if (config.error_callback)
                    config.error_callback(ErrorInfo{e.what()});
                if (++consecutive_errors >= 100) break;
                continue;
            }

            if (!have_packet) {
                try {
                    decoder->flush(pending);
                } catch (const std::runtime_error& e) {
                    if (config.error_policy == ErrorPolicy::THROW) throw;
                    error_count++;
                    if (config.error_callback)
                        config.error_callback(ErrorInfo{e.what()});
                }
                flushed = true;
                break;
            }

            try {
                decoder->decode(packet, pending);
                av_packet_unref(packet);
                consecutive_errors = 0;
            } catch (const std::runtime_error& e) {
                av_packet_unref(packet);
                if (config.error_policy == ErrorPolicy::THROW) throw;
                error_count++;
                if (config.error_callback)
                    config.error_callback(ErrorInfo{e.what()});
                decoder->reset();
                if (++consecutive_errors >= 100) break;
            }
            have = static_cast<int>(pending.size()) - pending_idx;
        }
        CUFRAME_NVTX_POP();
    }

    ~Impl() {
        // best-effort device restore for member destructors (no throw in dtor)
        cudaSetDevice(config.device_id);
        // release PooledBuffers before decoder dies
        pending.clear();
        for (auto* p : retained_nv12) cudaFree(p);
        for (auto* p : rgb_bufs) cudaFree(p);
        for (auto* p : out_bufs) cudaFree(p);
        if (batch_ready) cudaEventDestroy(batch_ready);
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

int Pipeline::source_width() const { return impl_->demuxer->video_info().width; }
int Pipeline::source_height() const { return impl_->demuxer->video_info().height; }
double Pipeline::fps() const { return impl_->demuxer->video_info().fps; }
int64_t Pipeline::frame_count() const { return impl_->demuxer->video_info().num_frames; }
const LetterboxInfo& Pipeline::letterbox_info() const { return impl_->letterbox; }
cudaStream_t Pipeline::stream() const { return impl_->preprocess_stream; }
cudaEvent_t Pipeline::batch_event() const { return impl_->batch_ready; }
size_t Pipeline::error_count() const { return impl_->error_count; }

void Pipeline::seek(double seconds) {
    auto& s = *impl_;
    CUFRAME_CUDA_CHECK(cudaSetDevice(s.config.device_id));

    // release all pending decoded frames (return PooledBuffers to decoder pool)
    s.pending.clear();
    s.pending_idx = 0;

    // reset decoder parser (clear NAL buffer, DPB, reordering state)
    s.decoder->reset();

    // seek demuxer to keyframe at or before target
    s.demuxer->seek(seconds);

    // reset pipeline state machine
    s.flushed = false;
    s.eos = false;
    s.retained_count_ = 0;

    // reset stride state — first frame after seek is always collected
    s.stride_skip = 0;

    // set up precise seek — next() will skip frames before target
    auto& info = s.demuxer->video_info();
    if (seconds <= 0.0) {
        s.seeking = false;
    } else {
        s.seeking = true;
        s.seek_target_pts = static_cast<int64_t>(
            seconds * info.time_base.den / info.time_base.num);
    }
}

void Pipeline::crop_rois(int batch_idx,
                          const Rect* rois, int num_rois,
                          GpuFrameBatch& output,
                          const NormParams& norm,
                          bool bgr) {
    auto& s = *impl_;
    if (!s.config.retain_decoded)
        throw std::logic_error("crop_rois requires retain_decoded(true)");
    if (batch_idx < 0 || batch_idx >= s.retained_count_)
        throw std::out_of_range("crop_rois: batch_idx out of range");

    auto& frame = s.retained_meta[batch_idx];
    roi_crop_batch(
        frame.data, frame.width, frame.height, frame.pitch,
        rois, num_rois,
        output.data(), output.width(), output.height(),
        s.config.color_matrix, norm, bgr,
        frame.is_10bit, s.preprocess_stream);

    CUFRAME_CUDA_CHECK(cudaStreamSynchronize(s.preprocess_stream));
    output.set_count(num_rois);
}

void Pipeline::crop_rois(int batch_idx,
                          const std::vector<Rect>& rois,
                          GpuFrameBatch& output,
                          const NormParams& norm,
                          bool bgr) {
    crop_rois(batch_idx, rois.data(), static_cast<int>(rois.size()), output, norm, bgr);
}

const RetainedFrame& Pipeline::retained_frame(int i) const {
    if (i < 0 || i >= impl_->retained_count_)
        throw std::out_of_range("retained_frame: index out of range");
    return impl_->retained_meta[i];
}

int Pipeline::retained_count() const {
    return impl_->retained_count_;
}

PipelineBuilder Pipeline::builder() { return PipelineBuilder{}; }

std::optional<std::shared_ptr<GpuFrameBatch>> Pipeline::next() {
    auto& s = *impl_;
    if (s.eos) return std::nullopt;

    CUFRAME_NVTX_PUSH("cuframe::next");
    CUFRAME_CUDA_CHECK(cudaSetDevice(s.config.device_id));

    auto batch = s.batch_pool->acquire();
    int collected = 0;

    while (collected < s.config.batch_size) {
        // ensure decoder D2D copies are visible to preprocess stream
        CUFRAME_CUDA_CHECK(cudaStreamWaitEvent(s.preprocess_stream,
                                                s.decoder->copy_done_event(), 0));

        // consume pending decoded frames
        while (s.pending_idx < static_cast<int>(s.pending.size())
               && collected < s.config.batch_size) {
            auto& frame = s.pending[s.pending_idx];

            // precise seek: skip frames before target timestamp
            if (s.seeking) {
                if (frame.timestamp < s.seek_target_pts) {
                    s.pending[s.pending_idx] = {};  // release PooledBuffer
                    s.pending_idx++;
                    continue;
                }
                s.seeking = false;
            }

            // temporal stride: skip frames between collected ones
            if (s.stride_skip > 0) {
                s.pending[s.pending_idx] = {};  // release PooledBuffer
                s.pending_idx++;
                s.stride_skip--;
                continue;
            }

            float* rgb = s.rgb_bufs.empty() ? nullptr : s.rgb_bufs[collected];
            s.preprocess(frame, s.out_bufs[collected], rgb, s.preprocess_stream);

            // retain NV12 copy before releasing pool buffer
            if (s.config.retain_decoded) {
                unsigned int luma_h = frame.height;
                unsigned int chroma_h = (frame.height + 1) / 2;
                size_t needed = frame.pitch * (luma_h + chroma_h);

                if (!s.retained_allocated) {
                    s.retained_frame_bytes = needed;
                    s.retained_nv12.resize(s.config.batch_size, nullptr);
                    for (int j = 0; j < s.config.batch_size; ++j)
                        CUFRAME_CUDA_CHECK(cudaMalloc(&s.retained_nv12[j],
                                                       s.retained_frame_bytes));
                    s.retained_meta.resize(s.config.batch_size);
                    s.retained_allocated = true;
                } else if (needed != s.retained_frame_bytes) {
                    throw std::runtime_error(
                        "retain_decoded: mid-stream resolution change "
                        "(frame needs " + std::to_string(needed) +
                        " bytes, buffer is " + std::to_string(s.retained_frame_bytes) + ")");
                }
                CUFRAME_CUDA_CHECK(cudaMemcpyAsync(
                    s.retained_nv12[collected], frame.buffer->data(),
                    s.retained_frame_bytes, cudaMemcpyDeviceToDevice,
                    s.preprocess_stream));
                s.retained_meta[collected] = {
                    s.retained_nv12[collected],
                    frame.width, frame.height, frame.pitch,
                    frame.bit_depth > 8
                };
            }

            s.pending[s.pending_idx] = {};  // release PooledBuffer to decoder pool
            s.pending_idx++;
            collected++;
            s.stride_skip = s.config.temporal_stride - 1;
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
        bool demuxer_done = false;
        int consecutive_errors = 0;
        while (!demuxer_done) {
            bool have_packet;
            try {
                have_packet = s.demuxer->read_packet(s.packet);
                consecutive_errors = 0;
            } catch (const std::runtime_error& e) {
                if (s.config.error_policy == ErrorPolicy::THROW) throw;
                s.error_count++;
                if (s.config.error_callback)
                    s.config.error_callback(ErrorInfo{e.what()});
                if (++consecutive_errors >= 100) { demuxer_done = true; break; }
                continue;
            }
            if (!have_packet) { demuxer_done = true; break; }

            try {
                int before = static_cast<int>(s.pending.size());
                s.decoder->decode(s.packet, s.pending);
                av_packet_unref(s.packet);
                consecutive_errors = 0;
                if (static_cast<int>(s.pending.size()) > before) {
                    got_frames = true;
                    break;
                }
            } catch (const std::runtime_error& e) {
                av_packet_unref(s.packet);
                if (s.config.error_policy == ErrorPolicy::THROW) throw;
                s.error_count++;
                if (s.config.error_callback)
                    s.config.error_callback(ErrorInfo{e.what()});
                // rebuild parser to clear corrupt state (loses DPB, up to 1 GOP)
                s.decoder->reset();
                if (++consecutive_errors >= 100) { demuxer_done = true; break; }
            }
        }

        if (!got_frames && demuxer_done) {
            try {
                s.decoder->flush(s.pending);
            } catch (const std::runtime_error& e) {
                if (s.config.error_policy == ErrorPolicy::THROW) throw;
                s.error_count++;
                if (s.config.error_callback)
                    s.config.error_callback(ErrorInfo{e.what()});
            }
            s.flushed = true;
        }
    }

    if (collected == 0) {
        s.eos = true;
        CUFRAME_NVTX_POP();
        return std::nullopt;
    }

    // copy preprocessed frames into contiguous batch
    for (int i = 0; i < collected; ++i)
        s.batch_ptrs[i] = s.out_bufs[i];
    batch_frames(*batch, s.batch_ptrs.data(), collected, s.preprocess_stream);

    // record event when batch GPU work completes
    CUFRAME_CUDA_CHECK(cudaEventRecord(s.batch_ready, s.preprocess_stream));

    // pre-decode next batch (CPU-side) while GPU finishes current batch
    s.prefetch();

    // wait for batch completion via event. more composable than stream sync —
    // downstream streams can cudaStreamWaitEvent on batch_ready without full sync.
    CUFRAME_CUDA_CHECK(cudaEventSynchronize(s.batch_ready));
    batch->set_count(collected);
    if (s.config.retain_decoded)
        s.retained_count_ = collected;
    CUFRAME_NVTX_POP();
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

PipelineBuilder& PipelineBuilder::normalize(const NormParams& params) {
    config_.has_normalize = true;
    config_.norm = params;
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

PipelineBuilder& PipelineBuilder::retain_decoded(bool retain) {
    config_.retain_decoded = retain;
    return *this;
}

PipelineBuilder& PipelineBuilder::device(int gpu_id) {
    config_.device_id = gpu_id;
    return *this;
}

PipelineBuilder& PipelineBuilder::temporal_stride(int stride) {
    if (stride < 1)
        throw std::invalid_argument("pipeline: temporal_stride must be >= 1");
    config_.temporal_stride = stride;
    return *this;
}

PipelineBuilder& PipelineBuilder::error_policy(ErrorPolicy policy) {
    config_.error_policy = policy;
    return *this;
}

PipelineBuilder& PipelineBuilder::on_error(ErrorCallback callback) {
    config_.error_callback = std::move(callback);
    return *this;
}

Pipeline PipelineBuilder::build() {
    if (config_.input_path.empty())
        throw std::invalid_argument("pipeline: input path required");
    if (config_.batch_size < 1)
        throw std::invalid_argument("pipeline: batch size must be >= 1");

    CUFRAME_CUDA_CHECK(cudaSetDevice(config_.device_id));

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

    // compute output→source coordinate mapping
    LetterboxInfo lb;
    if (config_.has_resize || config_.has_center_crop) {
        lb.scale_x = resize_params.scale_x;
        lb.scale_y = resize_params.scale_y;
        lb.pad_left = static_cast<float>(resize_params.pad_left);
        lb.pad_top = static_cast<float>(resize_params.pad_top);
        lb.offset_x = resize_params.src_offset_x;
        lb.offset_y = resize_params.src_offset_y;
    }

    bool use_fused = (config_.has_resize || config_.has_center_crop)
                     && config_.has_normalize;

    // pool needs room for prefetch buffer + in-flight batch + headroom.
    // with temporal stride, prefetch decodes stride*batch_size frames (capped at 4x).
    int prefetch_target = std::min(config_.batch_size * config_.temporal_stride,
                                   config_.batch_size * 4);
    int pool_count = prefetch_target + config_.batch_size + 4;
    auto decoder = std::make_unique<Decoder>(info, pool_count);

    cudaStream_t preprocess_stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&preprocess_stream));

    cudaEvent_t batch_ready;
    CUFRAME_CUDA_CHECK(cudaEventCreate(&batch_ready));

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
    impl->letterbox = lb;
    impl->use_fused = use_fused;
    impl->out_w = out_w;
    impl->out_h = out_h;
    impl->preprocess_stream = preprocess_stream;
    impl->batch_ready = batch_ready;
    impl->rgb_bufs = std::move(rgb_bufs);
    impl->out_bufs = std::move(out_bufs);
    impl->batch_ptrs.resize(impl->config.batch_size);
    impl->batch_pool = std::move(batch_pool);
    impl->pending.reserve(impl->config.batch_size * std::min(impl->config.temporal_stride, 4));
    impl->packet = av_packet_alloc();

    Pipeline p;
    p.impl_ = std::move(impl);
    return p;
}

} // namespace cuframe
