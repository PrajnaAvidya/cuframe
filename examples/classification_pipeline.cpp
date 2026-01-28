// classification pipeline: resize → center crop → ImageNet normalize.
//
// matches standard torchvision eval transforms for ResNet, EfficientNet, ViT, etc:
//   Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean, std)
//
// the resize + center crop are fused into a single kernel pass — no intermediate buffer.
//
// build: cmake --preset default && cmake --build build
// run:   ./build/examples/classification_pipeline <video_file>

#include "cuframe/pipeline.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(256, 256, cuframe::ResizeMode::STRETCH)
        .center_crop(224, 224)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(16)
        .build();

    printf("source: %dx%d, %.2f fps, %lld frames\n",
           pipeline.source_width(), pipeline.source_height(),
           pipeline.fps(), (long long)pipeline.frame_count());

    int total = 0;
    int batch_num = 0;

    while (auto batch = pipeline.next()) {
        auto& b = **batch;
        printf("batch %d: %d frames, shape: %dx%dx%d\n",
               batch_num++, b.count(), b.channels(), b.height(), b.width());

        // b.data() is a float* to contiguous NCHW 16x3x224x224 on GPU.
        // pass directly to inference — no copies needed.

        total += b.count();
    }

    printf("processed %d frames total\n", total);
    return 0;
}
