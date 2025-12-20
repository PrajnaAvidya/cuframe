#include "cuframe/demuxer.h"
#include "cuframe/decoder.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/packet.h>
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file> [frame_number] [output.nv12]\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];
    int target_frame = argc > 2 ? std::atoi(argv[2]) : 0;
    std::string out_path = argc > 3 ? argv[3] : "frame_dump.nv12";

    cuframe::Demuxer demuxer(path);
    auto& info = demuxer.video_info();
    printf("opened: %dx%d %s\n", info.width, info.height,
           avcodec_get_name(info.codec_id));

    cuframe::Decoder decoder(info, 20);
    AVPacket* pkt = av_packet_alloc();
    std::vector<cuframe::DecodedFrame> frames;
    int frame_count = 0;
    bool dumped = false;

    while (demuxer.read_packet(pkt) && !dumped) {
        decoder.decode(pkt, frames);
        av_packet_unref(pkt);

        for (auto& f : frames) {
            if (frame_count == target_frame) {
                int w = f.width;
                int h = f.height;
                unsigned int pitch = f.pitch;

                // copy from gpu
                std::vector<uint8_t> gpu_data(f.buffer->size());
                cudaMemcpy(gpu_data.data(), f.buffer->data(),
                           f.buffer->size(), cudaMemcpyDeviceToHost);

                // write packed NV12 (strip pitch padding)
                std::ofstream out(out_path, std::ios::binary);
                for (int row = 0; row < h; ++row)
                    out.write(reinterpret_cast<const char*>(
                        gpu_data.data() + row * pitch), w);

                int chroma_h = (h + 1) / 2;
                size_t chroma_offset = static_cast<size_t>(pitch) * h;
                for (int row = 0; row < chroma_h; ++row)
                    out.write(reinterpret_cast<const char*>(
                        gpu_data.data() + chroma_offset + row * pitch), w);

                printf("frame %d -> %s (%dx%d, %zu bytes)\n",
                       frame_count, out_path.c_str(), w, h,
                       static_cast<size_t>(w) * h * 3 / 2);
                printf("view: ffplay -f rawvideo -pixel_format nv12 -video_size %dx%d %s\n",
                       w, h, out_path.c_str());
                dumped = true;
                break;
            }
            frame_count++;
        }
        frames.clear();
    }

    if (!dumped) {
        // flush in case target frame is in the tail
        decoder.flush(frames);
        for (auto& f : frames) {
            if (frame_count == target_frame) {
                int w = f.width;
                int h = f.height;
                unsigned int pitch = f.pitch;

                std::vector<uint8_t> gpu_data(f.buffer->size());
                cudaMemcpy(gpu_data.data(), f.buffer->data(),
                           f.buffer->size(), cudaMemcpyDeviceToHost);

                std::ofstream out(out_path, std::ios::binary);
                for (int row = 0; row < h; ++row)
                    out.write(reinterpret_cast<const char*>(
                        gpu_data.data() + row * pitch), w);

                int chroma_h = (h + 1) / 2;
                size_t chroma_offset = static_cast<size_t>(pitch) * h;
                for (int row = 0; row < chroma_h; ++row)
                    out.write(reinterpret_cast<const char*>(
                        gpu_data.data() + chroma_offset + row * pitch), w);

                printf("frame %d -> %s (%dx%d, %zu bytes)\n",
                       frame_count, out_path.c_str(), w, h,
                       static_cast<size_t>(w) * h * 3 / 2);
                printf("view: ffplay -f rawvideo -pixel_format nv12 -video_size %dx%d %s\n",
                       w, h, out_path.c_str());
                dumped = true;
                break;
            }
            frame_count++;
        }
    }

    av_packet_free(&pkt);

    if (!dumped)
        fprintf(stderr, "frame %d not found (video has %d frames)\n",
                target_frame, frame_count);

    return dumped ? 0 : 1;
}
