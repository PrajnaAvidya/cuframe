#pragma once

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include "cuframe/frame_pool.h"
#include "cuframe/demuxer.h"

#include <memory>
#include <vector>
#include <cstdint>
#include <exception>
#include <cuda_runtime.h>

namespace cuframe {

struct PendingUnmap {
    CUdeviceptr ptr;
    cudaEvent_t event;
};

struct DecodedFrame {
    PooledBuffer buffer;    // borrows NV12/P016 data from pool, auto-returns
    int width = 0;
    int height = 0;
    unsigned int pitch = 0; // row stride in bytes (from cuvidMapVideoFrame)
    int64_t timestamp = 0;
    CUstream stream = nullptr;  // stream the frame data was produced on
    int bit_depth = 8;      // 8 for NV12, 10+ for P016
};

class Decoder {
public:
    explicit Decoder(const VideoInfo& info, int pool_count = 8);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // feed an encoded packet. any decoded frames are appended to out.
    // packet data must be in annex-b format (demuxer handles conversion).
    void decode(const AVPacket* packet, std::vector<DecodedFrame>& out);

    // flush remaining frames at end of stream
    void flush(std::vector<DecodedFrame>& out);

    // reset parser state after a seek. clears pending frames and recreates the
    // NVDEC parser so it can accept fresh packets from a new position.
    void reset();

    int width() const;
    int height() const;
    CUstream stream() const;

    // event recorded after each D2D copy in on_display.
    // downstream streams should cudaStreamWaitEvent on this before
    // reading decoded frame data.
    cudaEvent_t copy_done_event() const;

private:
    // static callback trampolines — cast user_data back to this
    static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* fmt);
    static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic);
    static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp);

    int on_sequence(CUVIDEOFORMAT* fmt);
    int on_decode(CUVIDPICPARAMS* pic);
    int on_display(CUVIDPARSERDISPINFO* disp);

    CUcontext cu_ctx_ = nullptr;
    CUstream stream_ = nullptr;
    CUvideoparser parser_ = nullptr;
    CUvideodecoder decoder_ = nullptr;
    CUVIDPARSERPARAMS parser_params_{};  // saved for reset

    int width_ = 0;
    int height_ = 0;
    unsigned int surface_height_ = 0;  // coded height (may differ from display)

    int pool_count_ = 8;
    int bit_depth_ = 8;
    std::unique_ptr<FramePool> pool_;
    std::vector<DecodedFrame> pending_frames_;

    // stored exception from callbacks — NVDEC callbacks are invoked from C code,
    // so throwing through them is UB. instead, catch and store here, rethrow after
    // cuvidParseVideoData returns.
    std::exception_ptr stored_error_;

    // deferred unmap — record event after D2D copy, unmap when event completes
    void drain_unmaps(bool force);
    cudaEvent_t acquire_event();

    std::vector<PendingUnmap> pending_unmaps_;
    std::vector<cudaEvent_t> event_pool_;
    int event_pool_idx_ = 0;
    cudaEvent_t copy_done_ = nullptr;  // latest D2D copy completion
};

// map ffmpeg codec id to nvdec codec enum. throws on unsupported codec.
cudaVideoCodec to_nvdec_codec(AVCodecID id);

} // namespace cuframe
