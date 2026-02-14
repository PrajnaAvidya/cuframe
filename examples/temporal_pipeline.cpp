#include <cuframe/cuframe.h>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <video>\n", argv[0]);
        return 1;
    }

    // video understanding: 16 frames sampled every 4th frame
    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(224, 224)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(16)
        .temporal_stride(4)
        .build();

    printf("source: %dx%d, %.1f fps, %lld frames\n",
           pipeline.source_width(), pipeline.source_height(),
           pipeline.fps(), pipeline.frame_count());

    int clip_idx = 0;
    while (auto clip = pipeline.next()) {
        // (*clip)->data() is float* to [count, 3, 224, 224] NCHW tensor on GPU
        // each frame is 4 decoded frames apart in time
        printf("clip %d: %d frames (spanning %d decoded frames)\n",
               clip_idx++, (*clip)->count(),
               (*clip)->count() * 4);
    }
    printf("total clips: %d\n", clip_idx);

    // random access: seek back to 1 second and grab a clip
    pipeline.seek(1.0);
    if (auto clip = pipeline.next()) {
        printf("after seek(1.0): clip with %d frames\n", (*clip)->count());
    }
}
