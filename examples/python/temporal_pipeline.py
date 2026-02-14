"""temporal stride + seek example.

video understanding models (SlowFast, TimeSformer, Video Swin, X3D) need
frames sampled across time, not consecutive frames. temporal_stride(N)
collects every Nth decoded frame into each batch.
"""
import cuframe

pipeline = (cuframe.Pipeline.builder()
    .input("video.mp4")
    .resize(224, 224)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .batch(16)
    .temporal_stride(4)  # every 4th frame
    .build())

print(f"source: {pipeline.source_width}x{pipeline.source_height}, "
      f"{pipeline.fps:.1f} fps, {pipeline.frame_count} frames")

# iterate clips — each batch has 16 frames spaced 4 apart
for i, clip in enumerate(pipeline):
    # torch.from_dlpack(clip) → [count, 3, 224, 224] CUDA tensor
    print(f"clip {i}: {clip.count} frames")

# random access: sample specific timestamps
for t in [1.0, 5.0, 10.0]:
    pipeline.seek(t)
    batch = pipeline.next()
    if batch is not None:
        # torch.from_dlpack(batch) for inference
        print(f"seek({t}): got {batch.count} frames")
    else:
        print(f"seek({t}): past end of video")
