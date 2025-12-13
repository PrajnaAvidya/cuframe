#pragma once

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include "cuframe/device_buffer.h"
#include "cuframe/demuxer.h"

#include <vector>
#include <cstdint>

namespace cuframe {

struct DecodedFrame {
    DeviceBuffer buffer;    // owns NV12 data in GPU memory
    int width = 0;
    int height = 0;
    unsigned int pitch = 0; // row stride in bytes (from cuvidMapVideoFrame)
    int64_t timestamp = 0;
};

class Decoder {
public:
    explicit Decoder(const VideoInfo& info);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // feed an encoded packet. any decoded frames are appended to out.
    // packet data must be in annex-b format (demuxer handles conversion).
    void decode(const AVPacket* packet, std::vector<DecodedFrame>& out);

    // flush remaining frames at end of stream
    void flush(std::vector<DecodedFrame>& out);

    int width() const;
    int height() const;

private:
    // static callback trampolines — cast user_data back to this
    static int CUDAAPI handle_sequence(void* user_data, CUVIDEOFORMAT* fmt);
    static int CUDAAPI handle_decode(void* user_data, CUVIDPICPARAMS* pic);
    static int CUDAAPI handle_display(void* user_data, CUVIDPARSERDISPINFO* disp);

    int on_sequence(CUVIDEOFORMAT* fmt);
    int on_decode(CUVIDPICPARAMS* pic);
    int on_display(CUVIDPARSERDISPINFO* disp);

    CUcontext cu_ctx_ = nullptr;
    CUvideoparser parser_ = nullptr;
    CUvideodecoder decoder_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    unsigned int surface_height_ = 0;  // coded height (may differ from display)

    std::vector<DecodedFrame> pending_frames_;
};

// map ffmpeg codec id to nvdec codec enum. throws on unsupported codec.
cudaVideoCodec to_nvdec_codec(AVCodecID id);

} // namespace cuframe
