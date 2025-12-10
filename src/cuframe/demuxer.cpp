#include "cuframe/demuxer.h"

#include <stdexcept>
#include <string>

extern "C" {
#include <libavutil/avutil.h>
}

namespace cuframe {

Demuxer::Demuxer(const std::string& filepath) {
    int ret = avformat_open_input(&fmt_ctx_, filepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw std::runtime_error("failed to open " + filepath + ": " + errbuf);
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("failed to find stream info in " + filepath);
    }

    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        avformat_close_input(&fmt_ctx_);
        throw std::runtime_error("no video stream found in " + filepath);
    }

    auto* codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;
    auto* stream = fmt_ctx_->streams[video_stream_idx_];

    info_.width = codecpar->width;
    info_.height = codecpar->height;
    info_.codec_id = codecpar->codec_id;
    info_.time_base = stream->time_base;
    info_.num_frames = stream->nb_frames > 0 ? stream->nb_frames : -1;

    if (codecpar->extradata && codecpar->extradata_size > 0) {
        info_.extradata.assign(
            codecpar->extradata,
            codecpar->extradata + codecpar->extradata_size
        );
    }
}

Demuxer::~Demuxer() {
    avformat_close_input(&fmt_ctx_);
}

const VideoInfo& Demuxer::video_info() const {
    return info_;
}

bool Demuxer::read_packet(AVPacket* packet) {
    while (true) {
        int ret = av_read_frame(fmt_ctx_, packet);
        if (ret == AVERROR_EOF) return false;
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error(std::string("av_read_frame error: ") + errbuf);
        }

        if (packet->stream_index == video_stream_idx_) return true;

        // not our video stream, skip
        av_packet_unref(packet);
    }
}

} // namespace cuframe
