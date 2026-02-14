// error_recovery.cpp — skip corrupt packets and continue decoding
//
// demonstrates ErrorPolicy::SKIP for processing potentially corrupt videos.
// errors are logged via on_error callback, processing continues at the next
// keyframe. check error_count() after iteration for a summary.

#include <cuframe/cuframe.h>
#include <cstdio>

int main() {
    auto pipeline = cuframe::Pipeline::builder()
        .input("video.mp4")
        .resize(640, 640)
        .normalize({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})  // YOLO: pixel / 255
        .batch(8)
        .error_policy(cuframe::ErrorPolicy::SKIP)
        .on_error([](const cuframe::ErrorInfo& info) {
            fprintf(stderr, "  skipped: %s\n", info.message.c_str());
        })
        .build();

    printf("source: %dx%d @ %.1f fps\n",
           pipeline.source_width(), pipeline.source_height(), pipeline.fps());

    int frames = 0;
    while (auto batch = pipeline.next()) {
        frames += (*batch)->count();
        // ... run inference on batch ...
    }

    printf("processed %d frames, %zu errors skipped\n",
           frames, pipeline.error_count());
    return 0;
}
