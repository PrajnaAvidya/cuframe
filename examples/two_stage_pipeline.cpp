// two-stage pipeline: detect objects, then crop + classify each detection.
//
// stage 1: decode → resize 640x640 → normalize → run detector (YOLO, etc.)
// stage 2: crop each detection box from the original NV12 frame → resize 224x224
//          → normalize → run classifier
//
// the key APIs: retain_decoded(true) keeps the raw NV12 frame alongside
// preprocessed output. Pipeline::crop_rois() crops all detections in a single
// kernel launch — handles frame lookup, color matrix, stream, and count
// automatically.
//
// detector and classifier calls are pseudocode. replace with real inference
// (TensorRT, ONNX Runtime, etc.) when integrating.
//
// build: cmake --preset default && cmake --build build
// run:   ./build/examples/two_stage_pipeline <video_file>

#include <cuframe/cuframe.h>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    // stage 1 pipeline: YOLO-style detection preprocessing.
    // retain_decoded(true) keeps NV12 frames for ROI cropping after detection.
    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
        .normalize({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})
        .retain_decoded(true)
        .batch(1)
        .build();

    printf("source: %dx%d, %.2f fps, %lld frames\n",
           pipeline.source_width(), pipeline.source_height(),
           pipeline.fps(), (long long)pipeline.frame_count());

    // crop pool for stage 2: 2 pool slots, up to 64 crops, 224x224 classifier input
    cuframe::BatchPool crop_pool(2, 64, 3, 224, 224);

    // reuse pipeline's norm params for the crop stage (same normalization).
    // use different NormParams if the classifier expects a different scheme.
    const auto& cfg = pipeline.config();

    int frame_num = 0;

    while (auto batch = pipeline.next()) {
        // ---------------------------------------------------------------
        // stage 1: detection (pseudocode)
        // ---------------------------------------------------------------
        // auto detections = run_detector((*batch)->data(), 640, 640);
        //
        // detector outputs boxes in 640x640 output space. use letterbox_info()
        // to map back to source pixel coordinates for ROI cropping:
        //
        //   auto& lb = pipeline.letterbox_info();
        //   for (auto& det : detections) {
        //       int sx = (int)lb.to_source_x(det.x);
        //       int sy = (int)lb.to_source_y(det.y);
        //       int sw = (int)(det.w * lb.scale_x);
        //       int sh = (int)(det.h * lb.scale_y);
        //       rois.push_back({sx, sy, sw, sh});
        //   }

        // simulate detector output: 2 detections in source pixel coordinates
        std::vector<cuframe::Rect> rois = {
            {100, 50, 200, 300},
            {400, 200, 150, 150}
        };

        if (rois.empty()) {
            frame_num++;
            continue;
        }

        // ---------------------------------------------------------------
        // stage 2: crop all detections → classify
        // ---------------------------------------------------------------

        auto crops = crop_pool.acquire();

        // single kernel launch: crop + resize + color convert + normalize all ROIs.
        // handles retained frame lookup, color matrix, stream, and set_count().
        pipeline.crop_rois(0, rois, *crops, cfg.norm);

        // crops->data() is contiguous NCHW: rois.size() x 3 x 224 x 224
        // pass directly to classifier — no copies needed.
        //
        // if the classifier supports dynamic batch (e.g. ONNX model exported
        // with dynamic=True), pass all crops in one inference call:
        //   auto labels = run_classifier(crops->data(), rois.size());
        //
        // otherwise loop per-crop:
        //   for (int i = 0; i < crops->count(); i++)
        //       auto label = run_classifier(crops->frame(i), 1);

        printf("frame %d: %d detections → %d crops (%dx%d)\n",
               frame_num, (int)rois.size(), crops->count(),
               crops->height(), crops->width());

        frame_num++;
    }

    printf("processed %d frames\n", frame_num);
    return 0;
}
