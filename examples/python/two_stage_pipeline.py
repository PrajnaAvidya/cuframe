"""two-stage pipeline: detect objects, crop + classify each detection.

stage 1: decode → resize 640x640 → normalize → run detector
stage 2: crop each detection box from the original frame → resize 224x224
         → normalize → run classifier

detector and classifier calls are pseudocode. replace with real inference
(PyTorch, ONNX Runtime, TensorRT) when integrating.

build: cmake --preset default && cmake --build build
run:   PYTHONPATH=python:build/src python examples/python/two_stage_pipeline.py <video_file>
"""
import sys
import cuframe

if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <video_file>")
    sys.exit(1)

# stage 1: YOLO-style detection preprocessing.
# retain_decoded(True) keeps NV12 frames for ROI cropping after detection.
pipeline = (cuframe.Pipeline.builder()
    .input(sys.argv[1])
    .resize(640, 640, cuframe.ResizeMode.LETTERBOX)
    .normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    .retain_decoded(True)
    .batch(1)
    .build())

print(f"source: {pipeline.source_width}x{pipeline.source_height}")

# stage 2: crop pool for classifier input (up to 64 crops at 224x224)
crop_pool = cuframe.BatchPool(1, 64, 3, 224, 224)

lb = pipeline.letterbox_info

for frame_num, batch in enumerate(pipeline):
    # --- run detector (placeholder) ---
    # tensor = torch.from_dlpack(batch)
    # detections = detector(tensor)

    # placeholder detections (in 640x640 output coords)
    det_boxes = [(100, 150, 50, 80), (300, 200, 60, 60)]

    # map output coords → source coords for ROI crop
    rois = []
    for dx, dy, dw, dh in det_boxes:
        sx = int(lb.to_source_x(dx))
        sy = int(lb.to_source_y(dy))
        sw = int(dw * lb.scale_x)
        sh = int(dh * lb.scale_y)
        rois.append((sx, sy, sw, sh))  # plain tuples work — no need for Rect()

    # crop + resize + color convert + normalize in one kernel launch.
    # also accepts cuframe.Rect objects or a mix of both.
    crops = crop_pool.acquire()
    pipeline.crop_rois(0, rois, crops, cuframe.IMAGENET_NORM)

    # zero-copy to framework:
    # crop_tensor = torch.from_dlpack(crops)
    # labels = classifier(crop_tensor)

    print(f"frame {frame_num}: {crops.count} ROIs → shape={crops.shape}")

    if frame_num >= 9:
        break

print("done")
