#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <string>
#include <vector>
#include <cstdint>

namespace cuframe {

struct VideoInfo {
    int width = 0;
    int height = 0;
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    AVRational time_base = {0, 1};
    int64_t num_frames = -1;            // -1 if unknown
    std::vector<uint8_t> extradata;     // codec config (SPS/PPS for h264, VPS/SPS/PPS for hevc)
};

class Demuxer {
public:
    explicit Demuxer(const std::string& filepath);
    ~Demuxer();

    Demuxer(const Demuxer&) = delete;
    Demuxer& operator=(const Demuxer&) = delete;

    const VideoInfo& video_info() const;

    // read next video packet. returns false at EOF.
    // caller owns packet lifecycle (av_packet_alloc/av_packet_free/av_packet_unref).
    bool read_packet(AVPacket* packet);

private:
    AVFormatContext* fmt_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    VideoInfo info_;
};

} // namespace cuframe
