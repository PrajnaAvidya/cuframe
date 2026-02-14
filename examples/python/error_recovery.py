"""error_recovery.py — skip corrupt packets and continue decoding"""

import cuframe

errors = []

pipeline = (cuframe.Pipeline.builder()
    .input("video.mp4")
    .resize(640, 640)
    .normalize(cuframe.YOLO_NORM)
    .batch(8)
    .error_policy(cuframe.ErrorPolicy.SKIP)
    .on_error(lambda info: errors.append(info.message))
    .build())

print(f"source: {pipeline.source_width}x{pipeline.source_height} @ {pipeline.fps:.1f} fps")

frames = 0
for batch in pipeline:
    frames += batch.count
    # ... run inference on batch ...

print(f"processed {frames} frames, {pipeline.error_count} errors skipped")
if errors:
    print(f"first error: {errors[0]}")
