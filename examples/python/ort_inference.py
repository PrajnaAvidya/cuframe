"""ONNX Runtime inference with cuframe — zero-copy GPU pipeline.

demonstrates the direct ORT io_binding path without torch as intermediary.
cuframe's GPU buffer binds directly to ORT via data_ptr — no copies anywhere.

requires: pip install onnxruntime-gpu

build: cmake --preset default && cmake --build build
run:   PYTHONPATH=python:build/src python examples/python/ort_inference.py <model.onnx> <video_file>
"""
import sys
import numpy as np
import cuframe

try:
    import onnxruntime as ort
except ImportError:
    print("error: onnxruntime not installed. run: pip install onnxruntime-gpu")
    sys.exit(1)

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} <model.onnx> <video_file>")
    sys.exit(1)

model_path, video_path = sys.argv[1], sys.argv[2]

# create ORT session with CUDA EP
opts = ort.SessionOptions()
opts.log_severity_level = 3
session = ort.InferenceSession(model_path, opts, providers=["CUDAExecutionProvider"])

# read model input dimensions
inp = session.get_inputs()[0]
_, _, model_h, model_w = inp.shape
input_name = inp.name
output_names = [o.name for o in session.get_outputs()]

# build pipeline matching model input
pipeline = (cuframe.Pipeline.builder()
    .input(video_path)
    .resize(model_w, model_h, cuframe.ResizeMode.LETTERBOX)
    .normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    .batch(1)
    .build())

print(f"model:  {model_path} ({model_w}x{model_h})")
print(f"source: {pipeline.source_width}x{pipeline.source_height} "
      f"@ {pipeline.fps:.1f} fps")

for i, batch in enumerate(pipeline):
    # bind cuframe's GPU buffer directly to ORT — no copies, no torch needed.
    # batch.data_ptr is the raw CUDA device pointer (int).
    # batch.shape is (count, channels, height, width).
    io = session.io_binding()
    io.bind_input(input_name, "cuda", 0, np.float32,
                  list(batch.shape), batch.data_ptr)
    for name in output_names:
        io.bind_output(name, "cuda")

    session.run_with_iobinding(io)

    outputs = io.copy_outputs_to_cpu()
    print(f"frame {i}: input={batch.shape}, output={outputs[0].shape}")

    if i >= 9:
        break

print("done")
