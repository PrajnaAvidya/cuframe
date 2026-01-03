#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/resize.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
#include <libswscale/swscale.h>
}

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using Clock = std::chrono::steady_clock;

static const int DST_W = 640;
static const int DST_H = 640;
static const int BATCH_SIZE = 8;

struct BenchResult {
    int frame_count = 0;
    double elapsed_ms = 0.0;
    double fps = 0.0;
};

// ============================================================================
// gpu pipeline: nvdec decode + separate kernels
// ============================================================================
static BenchResult bench_separate(const std::string& path) {
    cuframe::Demuxer demuxer(path);
    auto& info = demuxer.video_info();
    int src_w = info.width;
    int src_h = info.height;

    cuframe::Decoder decoder(info, 20);
    auto resize_params = cuframe::make_letterbox_params(src_w, src_h, DST_W, DST_H);

    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    // pre-allocate per-frame intermediate buffers
    size_t rgb_bytes = 3 * src_w * src_h * sizeof(float);
    size_t out_bytes = 3 * DST_W * DST_H * sizeof(float);

    std::vector<float*> rgb_bufs(BATCH_SIZE);
    std::vector<float*> out_bufs(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        CUFRAME_CUDA_CHECK(cudaMalloc(&rgb_bufs[i], rgb_bytes));
        CUFRAME_CUDA_CHECK(cudaMalloc(&out_bufs[i], out_bytes));
    }

    cuframe::GpuFrameBatch batch(BATCH_SIZE, 3, DST_H, DST_W);

    // decode + preprocess + batch
    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> pending;
    int total_frames = 0;
    int pending_idx = 0;

    auto t0 = Clock::now();

    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, pending);
        av_packet_unref(pkt);

        while (static_cast<int>(pending.size()) - pending_idx >= BATCH_SIZE) {
            std::vector<const float*> ptrs(BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                auto& f = pending[pending_idx + i];
                auto* nv12 = static_cast<const uint8_t*>(f.buffer->data());

                cuframe::nv12_to_rgb_planar(nv12, rgb_bufs[i],
                                             f.width, f.height, f.pitch,
                                             cuframe::BT601, stream);
                cuframe::resize_bilinear(rgb_bufs[i], out_bufs[i], resize_params, stream);
                cuframe::normalize(out_bufs[i], out_bufs[i], DST_W, DST_H,
                                    cuframe::IMAGENET_NORM, stream);
                ptrs[i] = out_bufs[i];
            }
            cuframe::batch_frames(batch, ptrs.data(), BATCH_SIZE, stream);
            CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));
            // release consumed frames to return pool buffers
            for (int i = 0; i < BATCH_SIZE; ++i)
                pending[pending_idx + i] = {};
            total_frames += BATCH_SIZE;
            pending_idx += BATCH_SIZE;
        }
    }

    // flush decoder
    decoder.flush(pending);

    // process remaining frames
    int remaining = static_cast<int>(pending.size()) - pending_idx;
    if (remaining > 0) {
        int batch_n = std::min(remaining, BATCH_SIZE);
        std::vector<const float*> ptrs(batch_n);
        for (int i = 0; i < batch_n; ++i) {
            auto& f = pending[pending_idx + i];
            auto* nv12 = static_cast<const uint8_t*>(f.buffer->data());

            cuframe::nv12_to_rgb_planar(nv12, rgb_bufs[i],
                                         f.width, f.height, f.pitch,
                                         cuframe::BT601, stream);
            cuframe::resize_bilinear(rgb_bufs[i], out_bufs[i], resize_params, stream);
            cuframe::normalize(out_bufs[i], out_bufs[i], DST_W, DST_H,
                                cuframe::IMAGENET_NORM, stream);
            ptrs[i] = out_bufs[i];
        }
        cuframe::batch_frames(batch, ptrs.data(), batch_n, stream);
        CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));
        total_frames += batch_n;
    }

    auto t1 = Clock::now();

    av_packet_free(&pkt);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        cudaFree(rgb_bufs[i]);
        cudaFree(out_bufs[i]);
    }
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total_frames, ms, total_frames / (ms / 1000.0)};
}

// ============================================================================
// gpu pipeline: nvdec decode + fused kernel
// ============================================================================
static BenchResult bench_fused(const std::string& path) {
    cuframe::Demuxer demuxer(path);
    auto& info = demuxer.video_info();
    int src_w = info.width;
    int src_h = info.height;

    cuframe::Decoder decoder(info, 20);
    auto resize_params = cuframe::make_letterbox_params(src_w, src_h, DST_W, DST_H);

    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    size_t out_bytes = 3 * DST_W * DST_H * sizeof(float);
    std::vector<float*> out_bufs(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i)
        CUFRAME_CUDA_CHECK(cudaMalloc(&out_bufs[i], out_bytes));

    cuframe::GpuFrameBatch batch(BATCH_SIZE, 3, DST_H, DST_W);

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> pending;
    int total_frames = 0;
    int pending_idx = 0;

    auto t0 = Clock::now();

    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, pending);
        av_packet_unref(pkt);

        while (static_cast<int>(pending.size()) - pending_idx >= BATCH_SIZE) {
            std::vector<const float*> ptrs(BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                auto& f = pending[pending_idx + i];
                auto* nv12 = static_cast<const uint8_t*>(f.buffer->data());

                cuframe::fused_nv12_to_tensor(nv12, out_bufs[i],
                                               f.width, f.height, f.pitch,
                                               resize_params, cuframe::BT601,
                                               cuframe::IMAGENET_NORM, stream);
                ptrs[i] = out_bufs[i];
            }
            cuframe::batch_frames(batch, ptrs.data(), BATCH_SIZE, stream);
            CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < BATCH_SIZE; ++i)
                pending[pending_idx + i] = {};
            total_frames += BATCH_SIZE;
            pending_idx += BATCH_SIZE;
        }
    }

    decoder.flush(pending);

    int remaining = static_cast<int>(pending.size()) - pending_idx;
    if (remaining > 0) {
        int n = std::min(remaining, BATCH_SIZE);
        std::vector<const float*> ptrs(n);
        for (int i = 0; i < n; ++i) {
            auto& f = pending[pending_idx + i];
            auto* nv12 = static_cast<const uint8_t*>(f.buffer->data());

            cuframe::fused_nv12_to_tensor(nv12, out_bufs[i],
                                           f.width, f.height, f.pitch,
                                           resize_params, cuframe::BT601,
                                           cuframe::IMAGENET_NORM, stream);
            ptrs[i] = out_bufs[i];
        }
        cuframe::batch_frames(batch, ptrs.data(), n, stream);
        CUFRAME_CUDA_CHECK(cudaStreamSynchronize(stream));
        total_frames += n;
    }

    auto t1 = Clock::now();

    pending.clear();
    av_packet_free(&pkt);
    for (int i = 0; i < BATCH_SIZE; ++i)
        cudaFree(out_bufs[i]);
    CUFRAME_CUDA_CHECK(cudaStreamDestroy(stream));

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total_frames, ms, total_frames / (ms / 1000.0)};
}

// ============================================================================
// cpu baseline: ffmpeg software decode + sws_scale + manual normalize + cudaMemcpy
// ============================================================================
static BenchResult bench_cpu_baseline(const std::string& path) {
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

    int src_w = ctx->width;
    int src_h = ctx->height;

    // sws: NV12 → RGB24 at target resolution
    SwsContext* sws = sws_getContext(src_w, src_h, AV_PIX_FMT_YUV420P,
                                      DST_W, DST_H, AV_PIX_FMT_RGB24,
                                      SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws) {
        fprintf(stderr, "cpu: sws_getContext failed\n");
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
        return {};
    }

    // host buffers
    std::vector<uint8_t> rgb24_buf(DST_W * DST_H * 3);
    size_t frame_floats = 3 * DST_W * DST_H;
    size_t frame_bytes = frame_floats * sizeof(float);
    std::vector<float> float_buf(frame_floats);

    // gpu batch
    cuframe::GpuFrameBatch batch(BATCH_SIZE, 3, DST_H, DST_W);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int total_frames = 0;
    int batch_idx = 0;

    auto process_frame = [&]() {
        // sws_scale: yuv420p → RGB24 + resize
        uint8_t* dst_data[1] = { rgb24_buf.data() };
        int dst_linesize[1] = { DST_W * 3 };
        sws_scale(sws, frame->data, frame->linesize, 0, src_h,
                  dst_data, dst_linesize);

        // convert packed uint8 RGB24 → float32 planar + normalize
        int plane = DST_W * DST_H;
        for (int y = 0; y < DST_H; ++y) {
            for (int x = 0; x < DST_W; ++x) {
                int src_idx = (y * DST_W + x) * 3;
                int pixel = y * DST_W + x;
                for (int c = 0; c < 3; ++c) {
                    float val = static_cast<float>(rgb24_buf[src_idx + c]);
                    float_buf[c * plane + pixel] = val * cuframe::IMAGENET_NORM.scale[c]
                                                 + cuframe::IMAGENET_NORM.bias[c];
                }
            }
        }

        // copy to gpu
        CUFRAME_CUDA_CHECK(cudaMemcpy(batch.frame(batch_idx), float_buf.data(),
                                       frame_bytes, cudaMemcpyHostToDevice));

        total_frames++;
        batch_idx++;

        // sync at batch boundary (simulate inference)
        if (batch_idx >= BATCH_SIZE) {
            CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());
            batch_idx = 0;
        }
    };

    auto t0 = Clock::now();

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != stream_idx) {
            av_packet_unref(pkt);
            continue;
        }
        avcodec_send_packet(ctx, pkt);
        while (avcodec_receive_frame(ctx, frame) == 0)
            process_frame();
        av_packet_unref(pkt);
    }
    // flush decoder
    avcodec_send_packet(ctx, nullptr);
    while (avcodec_receive_frame(ctx, frame) == 0)
        process_frame();

    // final partial batch sync
    if (batch_idx > 0)
        CUFRAME_CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = Clock::now();

    av_frame_free(&frame);
    av_packet_free(&pkt);
    sws_freeContext(sws);
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total_frames, ms, total_frames / (ms / 1000.0)};
}

// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    av_log_set_level(AV_LOG_FATAL);

    // probe video info
    cuframe::Demuxer probe(path);
    auto& info = probe.video_info();
    const char* codec_name = avcodec_get_name(info.codec_id);
    printf("preprocessing benchmark: %dx%d (%s) -> %dx%d, batch size %d, ImageNet norm\n\n",
           info.width, info.height, codec_name, DST_W, DST_H, BATCH_SIZE);

    // warm up gpu
    cudaFree(0);

    auto separate = bench_separate(path);
    printf("nvdec + separate (colorconv + resize + norm + batch):\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           separate.frame_count, separate.elapsed_ms, separate.fps);

    auto fused = bench_fused(path);
    printf("nvdec + fused (fused_nv12_to_tensor + batch):\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           fused.frame_count, fused.elapsed_ms, fused.fps);

    auto cpu = bench_cpu_baseline(path);
    printf("cpu baseline (ffmpeg decode + sws_scale + normalize + cudaMemcpy):\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           cpu.frame_count, cpu.elapsed_ms, cpu.fps);

    // speedup summary
    if (cpu.fps > 0) {
        printf("speedup vs cpu: separate %.1fx, fused %.1fx\n",
               separate.fps / cpu.fps, fused.fps / cpu.fps);
    }
    if (separate.fps > 0) {
        printf("fused vs separate: %.2fx\n", fused.fps / separate.fps);
    }

    // gpu memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used = total_mem - free_mem;
    printf("\ngpu memory: %zu MB used / %zu MB total\n",
           used / (1024 * 1024), total_mem / (1024 * 1024));

    return 0;
}
