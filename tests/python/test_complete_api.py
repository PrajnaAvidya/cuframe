import cuframe
import pytest

VIDEO = "tests/data/test_h264.mp4"


def test_norm_params():
    norm = cuframe.make_norm_params([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert len(norm.scale) == 3
    assert len(norm.bias) == 3
    assert "NormParams" in repr(norm)


def test_norm_constants():
    assert hasattr(cuframe, "IMAGENET_NORM")
    assert hasattr(cuframe, "YOLO_NORM")
    yn = cuframe.YOLO_NORM
    # YOLO_NORM: scale = 1/255 per channel, bias = 0
    assert abs(yn.scale[0] - 1.0 / 255.0) < 1e-6
    assert abs(yn.bias[0]) < 1e-6


def test_rect():
    r = cuframe.Rect(10, 20, 100, 50)
    assert r.x == 10
    assert r.y == 20
    assert r.w == 100
    assert r.h == 50
    assert "Rect" in repr(r)

    # read/write
    r.x = 42
    assert r.x == 42


def test_letterbox_info():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640, cuframe.ResizeMode.LETTERBOX) \
        .batch(1) \
        .build()

    lb = pipeline.letterbox_info
    assert lb.scale_x > 0
    assert lb.scale_y > 0
    assert "LetterboxInfo" in repr(lb)

    # 320x240 source → 640x640 letterbox
    # width-limited: scale = 640/320 = 2.0, source per output = 320/640 = 0.5
    assert abs(lb.scale_x - 0.5) < 0.01

    # round-trip: source → output → source
    src_x = 100.0
    out_x = src_x / lb.scale_x + lb.pad_left
    assert abs(lb.to_source_x(out_x) - src_x) < 0.01


def test_batch_pool():
    pool = cuframe.BatchPool(2, 4, 3, 224, 224)
    assert pool.capacity == 2
    assert pool.available == 2

    batch = pool.acquire()
    assert pool.available == 1
    assert batch.height == 224
    assert batch.width == 224

    del batch
    import gc; gc.collect()
    assert pool.available == 2


def test_batch_pool_try_acquire():
    pool = cuframe.BatchPool(1, 4, 3, 64, 64)
    b1 = pool.acquire()
    assert pool.available == 0

    b2 = pool.try_acquire()
    assert b2 is None

    del b1
    import gc; gc.collect()
    b3 = pool.try_acquire()
    assert b3 is not None
    assert b3.height == 64


def test_crop_rois():
    torch = pytest.importorskip("torch")

    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640) \
        .normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) \
        .retain_decoded(True) \
        .batch(1) \
        .build()

    batch = next(iter(pipeline))

    pool = cuframe.BatchPool(1, 4, 3, 224, 224)
    crops = pool.acquire()

    rois = [cuframe.Rect(10, 10, 100, 80), cuframe.Rect(50, 30, 60, 60)]
    pipeline.crop_rois(0, rois, crops, cuframe.YOLO_NORM)

    assert crops.count == 2
    tensor = torch.from_dlpack(crops)
    assert tensor.shape == (2, 3, 224, 224)
    assert tensor.device.type == "cuda"


def test_crop_rois_error():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640) \
        .batch(1) \
        .build()

    next(iter(pipeline))

    pool = cuframe.BatchPool(1, 4, 3, 224, 224)
    crops = pool.acquire()

    # retain_decoded not enabled — should throw
    with pytest.raises(Exception):
        pipeline.crop_rois(0, [cuframe.Rect(0, 0, 10, 10)], crops, cuframe.YOLO_NORM)


def test_center_crop():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(256, 256, cuframe.ResizeMode.STRETCH) \
        .center_crop(224, 224) \
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        .batch(4) \
        .build()

    batch = next(iter(pipeline))
    assert batch.height == 224
    assert batch.width == 224
