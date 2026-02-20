#include "cuframe/demuxer.h"

#include <cstring>
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
    if (stream->avg_frame_rate.den > 0)
        info_.fps = (double)stream->avg_frame_rate.num / stream->avg_frame_rate.den;
    info_.num_frames = stream->nb_frames > 0 ? stream->nb_frames : -1;

    if (codecpar->extradata && codecpar->extradata_size > 0) {
        info_.extradata.assign(
            codecpar->extradata,
            codecpar->extradata + codecpar->extradata_size
        );
    }

    // nvdec parser expects annex-b format (start code prefixed NALUs).
    // mp4/mkv/flv containers store h264/hevc in AVCC format (length prefixed).
    // apply bitstream filter to convert.
    const char* fmt_name = fmt_ctx_->iformat->name;
    // canonical: "mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "flv"
    bool is_mp4_like = (strstr(fmt_name, "mov") || strstr(fmt_name, "matroska")
                        || strstr(fmt_name, "flv"));

    const char* bsf_name = nullptr;
    if (is_mp4_like && info_.codec_id == AV_CODEC_ID_H264)
        bsf_name = "h264_mp4toannexb";
    else if (is_mp4_like && info_.codec_id == AV_CODEC_ID_HEVC)
        bsf_name = "hevc_mp4toannexb";

    if (bsf_name) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsf_name);
        if (!bsf) throw std::runtime_error(std::string(bsf_name) + " bsf not found");
        av_bsf_alloc(bsf, &bsf_ctx_);
        avcodec_parameters_copy(bsf_ctx_->par_in, codecpar);
        av_bsf_init(bsf_ctx_);
    }
}

Demuxer::~Demuxer() {
    if (bsf_ctx_) av_bsf_free(&bsf_ctx_);
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

        if (packet->stream_index == video_stream_idx_) {
            if (bsf_ctx_) {
                // convert AVCC -> annex B
                int bsf_ret = av_bsf_send_packet(bsf_ctx_, packet);
                av_packet_unref(packet);
                if (bsf_ret < 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(bsf_ret, errbuf, sizeof(errbuf));
                    throw std::runtime_error(
                        std::string("av_bsf_send_packet: ") + errbuf);
                }

                bsf_ret = av_bsf_receive_packet(bsf_ctx_, packet);
                if (bsf_ret == AVERROR(EAGAIN))
                    continue;
                if (bsf_ret < 0) {
                    char errbuf[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(bsf_ret, errbuf, sizeof(errbuf));
                    throw std::runtime_error(
                        std::string("av_bsf_receive_packet: ") + errbuf);
                }
            }
            return true;
        }

        // not our video stream, skip
        av_packet_unref(packet);
    }
}

void Demuxer::seek(double seconds) {
    auto* stream = fmt_ctx_->streams[video_stream_idx_];
    int64_t ts = (seconds <= 0.0) ? 0
        : static_cast<int64_t>(seconds * stream->time_base.den / stream->time_base.num);

    int ret = av_seek_frame(fmt_ctx_, video_stream_idx_, ts, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw std::runtime_error("seek failed: " + std::string(errbuf));
    }

    // flush bitstream filter state (stale annex-b conversion context)
    if (bsf_ctx_)
        av_bsf_flush(bsf_ctx_);
}

} // namespace cuframe
