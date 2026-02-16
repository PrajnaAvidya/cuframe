#include "cuframe/pipeline.h"
#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include "cuframe/kernels/fused_preprocess.h"
#include "cuframe/kernels/color_convert.h"
#include "cuframe/kernels/normalize.h"
#include "cuframe/gpu_frame_batch.h"
#include "cuframe/cuda_utils.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
}

#include <chrono>
#include <cstdio>
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
// pipeline API (with multi-stream prefetch)
// ============================================================================
static BenchResult bench_pipeline(const std::string& path) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(path)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(BATCH_SIZE)
        .build();

    int total = 0;
    auto t0 = Clock::now();

    while (auto batch = pipeline.next())
        total += (*batch)->count();

    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total, ms, total / (ms / 1000.0)};
}

// ============================================================================
// pipeline API — GPU event timing (measures GPU work only, no CPU overhead)
// ============================================================================
static BenchResult bench_pipeline_gpu_timed(const std::string& path) {
    auto pipeline = cuframe::Pipeline::builder()
        .input(path)
        .resize(DST_W, DST_H)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(BATCH_SIZE)
        .build();

    cudaEvent_t start, stop;
    CUFRAME_CUDA_CHECK(cudaEventCreate(&start));
    CUFRAME_CUDA_CHECK(cudaEventCreate(&stop));

    int total = 0;
    CUFRAME_CUDA_CHECK(cudaEventRecord(start, pipeline.stream()));

    while (auto batch = pipeline.next())
        total += (*batch)->count();

    CUFRAME_CUDA_CHECK(cudaEventRecord(stop, pipeline.stream()));
    CUFRAME_CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUFRAME_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    CUFRAME_CUDA_CHECK(cudaEventDestroy(start));
    CUFRAME_CUDA_CHECK(cudaEventDestroy(stop));

    return {total, static_cast<double>(gpu_ms), total / (gpu_ms / 1000.0)};
}

// ============================================================================
// raw fused kernels (no pipeline abstraction)
// ============================================================================
static BenchResult bench_raw_fused(const std::string& path) {
    cuframe::Demuxer demuxer(path);
    auto& info = demuxer.video_info();

    cuframe::Decoder decoder(info, 20);
    auto resize_params = cuframe::make_letterbox_params(
        info.width, info.height, DST_W, DST_H);
    auto color = (info.height > 720) ? cuframe::BT709 : cuframe::BT601;

    cudaStream_t stream;
    CUFRAME_CUDA_CHECK(cudaStreamCreate(&stream));

    size_t out_bytes = 3ULL * DST_W * DST_H * sizeof(float);
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
                    resize_params, color, cuframe::IMAGENET_NORM, stream);
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
                resize_params, color, cuframe::IMAGENET_NORM, stream);
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
// decode only (no preprocessing — measures decode ceiling)
// ============================================================================
static BenchResult bench_decode_only(const std::string& path) {
    cuframe::Demuxer demuxer(path);
    auto& info = demuxer.video_info();
    cuframe::Decoder decoder(info, 20);

    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> pending;
    int total = 0;
    int pending_idx = 0;

    auto t0 = Clock::now();

    while (demuxer.read_packet(pkt)) {
        decoder.decode(pkt, pending);
        av_packet_unref(pkt);

        // release decoded frames to keep pool from exhausting
        while (pending_idx < static_cast<int>(pending.size())) {
            pending[pending_idx] = {};
            pending_idx++;
            total++;
        }
    }

    decoder.flush(pending);
    while (pending_idx < static_cast<int>(pending.size())) {
        pending[pending_idx] = {};
        pending_idx++;
        total++;
    }

    auto t1 = Clock::now();

    av_packet_free(&pkt);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {total, ms, total / (ms / 1000.0)};
}

// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    av_log_set_level(AV_LOG_FATAL);

    cuframe::Demuxer probe(path);
    auto& info = probe.video_info();
    const char* codec_name = avcodec_get_name(info.codec_id);
    printf("pipeline benchmark: %dx%d (%s) -> %dx%d, batch size %d, ImageNet norm\n\n",
           info.width, info.height, codec_name, DST_W, DST_H, BATCH_SIZE);

    cudaFree(0);  // warm up gpu

    auto pipe = bench_pipeline(path);
    printf("pipeline API (multi-stream prefetch) [wall clock]:\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           pipe.frame_count, pipe.elapsed_ms, pipe.fps);

    auto pipe_gpu = bench_pipeline_gpu_timed(path);
    printf("pipeline API (multi-stream prefetch) [GPU events]:\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           pipe_gpu.frame_count, pipe_gpu.elapsed_ms, pipe_gpu.fps);

    auto raw = bench_raw_fused(path);
    printf("raw fused kernels (no pipeline):\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           raw.frame_count, raw.elapsed_ms, raw.fps);

    auto decode = bench_decode_only(path);
    printf("decode only (no preprocessing):\n");
    printf("  %d frames in %.1fms  (%.0f fps)\n\n",
           decode.frame_count, decode.elapsed_ms, decode.fps);

    // summary
    if (raw.fps > 0)
        printf("pipeline overhead vs raw: %.1f%%\n",
               (raw.fps / pipe.fps - 1.0) * 100.0);
    if (decode.fps > 0)
        printf("preprocessing cost: decode %.0f fps -> pipeline %.0f fps (%.1f%% overhead)\n",
               decode.fps, pipe.fps, (decode.fps / pipe.fps - 1.0) * 100.0);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used = total_mem - free_mem;
    printf("\ngpu memory: %zu MB used / %zu MB total\n",
           used / (1024 * 1024), total_mem / (1024 * 1024));

    return 0;
}
