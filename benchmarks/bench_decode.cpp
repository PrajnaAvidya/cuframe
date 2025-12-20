#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>
}

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using Clock = std::chrono::steady_clock;

struct BenchResult {
    int frame_count = 0;
    double elapsed_ms = 0.0;
    double fps = 0.0;
};

// hardware decode via our library
static BenchResult bench_nvdec(const std::string& path) {
    cuframe::Demuxer demuxer(path);
    cuframe::Decoder decoder(demuxer.video_info(), 20);

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> frames;
    int count = 0;

    auto t0 = Clock::now();
    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, frames);
        count += static_cast<int>(frames.size());
        frames.clear();  // return pooled buffers
        av_packet_unref(pkt);
    }
    decoder.flush(frames);
    count += static_cast<int>(frames.size());
    frames.clear();
    auto t1 = Clock::now();

    av_packet_free(&pkt);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {count, ms, count / (ms / 1000.0)};
}

// cpu software decode via ffmpeg
static BenchResult bench_cpu(const std::string& path) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "cpu: failed to open %s\n", path.c_str());
        return {};
    }
    avformat_find_stream_info(fmt, nullptr);

    int stream_idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_idx < 0) {
        fprintf(stderr, "cpu: no video stream found\n");
        avformat_close_input(&fmt);
        return {};
    }

    auto* par = fmt->streams[stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    AVCodecContext* ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(ctx, par);
    avcodec_open2(ctx, codec, nullptr);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int count = 0;

    auto t0 = Clock::now();
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != stream_idx) {
            av_packet_unref(pkt);
            continue;
        }
        avcodec_send_packet(ctx, pkt);
        while (avcodec_receive_frame(ctx, frame) == 0)
            count++;
        av_packet_unref(pkt);
    }
    // flush
    avcodec_send_packet(ctx, nullptr);
    while (avcodec_receive_frame(ctx, frame) == 0)
        count++;
    auto t1 = Clock::now();

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {count, ms, count / (ms / 1000.0)};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];

    // suppress ffmpeg log noise (eg sps_id errors from blu-ray 3D remuxes)
    av_log_set_level(AV_LOG_FATAL);

    // probe video info for display
    cuframe::Demuxer probe(path);
    auto& info = probe.video_info();
    const char* codec_name = avcodec_get_name(info.codec_id);
    printf("decoding: %s\n", path.c_str());
    printf("  %dx%d, %s", info.width, info.height, codec_name);
    if (info.num_frames > 0)
        printf(", %lld frames (reported)\n", static_cast<long long>(info.num_frames));
    else
        printf("\n");

    // warm up gpu
    cudaFree(0);

    printf("\n");

    auto nvdec = bench_nvdec(path);
    printf("nvdec:  %d frames in %.1fms  (%.0f fps)\n",
           nvdec.frame_count, nvdec.elapsed_ms, nvdec.fps);

    auto cpu = bench_cpu(path);
    printf("cpu:    %d frames in %.1fms  (%.0f fps)\n",
           cpu.frame_count, cpu.elapsed_ms, cpu.fps);

    if (cpu.fps > 0)
        printf("speedup: %.1fx\n", nvdec.fps / cpu.fps);

    // gpu memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("\ngpu memory: %zu MB free / %zu MB total\n",
           free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    return 0;
}
