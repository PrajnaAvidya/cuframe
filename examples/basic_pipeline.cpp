// basic cuframe pipeline usage: video file → batched NCHW tensors on GPU.
//
// build: cmake --preset default && cmake --build build
// run:   ./build/examples/basic_pipeline <video_file>

#include "cuframe/pipeline.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    // build a pipeline: decode → resize 640x640 (letterbox) → ImageNet normalize → batch 8
    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total = 0;
    int batch_num = 0;

    // pull batches until end of stream
    while (auto batch = pipeline.next()) {
        auto& b = **batch;
        printf("batch %d: %d frames, shape: %dx%dx%d, device ptr: %p\n",
               batch_num++, b.count(), b.channels(), b.height(), b.width(),
               static_cast<void*>(b.data()));

        // b.data() is a float* to a contiguous NCHW tensor in GPU memory.
        // pass it directly to TensorRT, ONNX Runtime, or any CUDA-aware
        // inference framework — no copies needed.

        total += b.count();
    }

    printf("processed %d frames total\n", total);
    return 0;
}
