"""minimal cuframe pipeline — decode, preprocess, iterate on GPU.

build: cmake --preset default && cmake --build build
run:   PYTHONPATH=python:build/src python examples/python/basic_pipeline.py <video_file>

or after `pip install .`:
       python examples/python/basic_pipeline.py <video_file>
"""
import sys
import cuframe

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <video_file>")
    sys.exit(1)

# decode → resize 640x640 (letterbox) → ImageNet normalize → batch 8
pipeline = (cuframe.Pipeline.builder()
    .input(sys.argv[1])
    .resize(640, 640, cuframe.ResizeMode.LETTERBOX)
    .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    .batch(8)
    .build())

print(f"source: {pipeline.source_width}x{pipeline.source_height} "
      f"@ {pipeline.fps:.1f} fps")

# frame_count may be -1 for containers that don't store it (MKV, some WebM).
# always iterate with `for batch in pipeline:` — never `range(frame_count)`.
if pipeline.frame_count >= 0:
    print(f"frames: {pipeline.frame_count}")
else:
    print("frames: unknown (container doesn't report count)")

total = 0
for i, batch in enumerate(pipeline):
    print(f"batch {i}: {batch.count} frames, shape={batch.shape}")
    total += batch.count

    # zero-copy to PyTorch:
    # import torch
    # tensor = torch.from_dlpack(batch)  # (N, 3, 640, 640) float32 cuda
    # output = model(tensor)

    # zero-copy to ONNX Runtime (no torch dependency needed):
    # import numpy as np
    # io = session.io_binding()
    # io.bind_input("images", "cuda", 0, np.float32,
    #               list(batch.shape), batch.data_ptr)
    # io.bind_output("output0", "cuda")
    # session.run_with_iobinding(io)

print(f"processed {total} frames total")
