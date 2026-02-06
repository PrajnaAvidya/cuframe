import cuframe
import pytest

VIDEO = "tests/data/test_h264.mp4"


def test_import():
    assert hasattr(cuframe, "Pipeline")
    assert hasattr(cuframe, "ResizeMode")


def test_resize_mode_enum():
    assert cuframe.ResizeMode.LETTERBOX != cuframe.ResizeMode.STRETCH


def test_pipeline_basic():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640) \
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        .batch(8) \
        .build()

    count = 0
    for batch in pipeline:
        assert batch.channels == 3
        assert batch.height == 640
        assert batch.width == 640
        assert batch.count <= batch.batch_size
        count += 1
    assert count > 0


def test_pipeline_next():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(224, 224, cuframe.ResizeMode.STRETCH) \
        .batch(4) \
        .build()

    batch = pipeline.next()
    assert batch is not None
    assert batch.height == 224
    assert batch.width == 224

    # drain
    while pipeline.next() is not None:
        pass

    # should return None at end
    assert pipeline.next() is None


def test_pipeline_metadata():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .batch(1) \
        .build()

    assert pipeline.source_width == 320
    assert pipeline.source_height == 240
    assert pipeline.fps > 0
    assert pipeline.frame_count > 0


def test_pipeline_error():
    with pytest.raises(Exception):
        cuframe.Pipeline.builder().build()
    with pytest.raises(Exception):
        cuframe.Pipeline.builder().input("nonexistent.mp4").build()


def test_batch_repr():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(4) \
        .build()

    batch = next(iter(pipeline))
    r = repr(batch)
    assert "GpuFrameBatch" in r
    assert "320" in r


def test_dlpack_torch():
    torch = pytest.importorskip("torch")

    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640) \
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        .batch(8) \
        .build()

    batch = next(iter(pipeline))
    tensor = torch.from_dlpack(batch)

    assert tensor.shape == (batch.count, 3, 640, 640)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cuda"


def test_dlpack_lifetime():
    """batch memory stays alive as long as torch tensor references it"""
    torch = pytest.importorskip("torch")

    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(640, 640) \
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        .batch(8) \
        .pool_size(1) \
        .build()

    batch = next(iter(pipeline))
    tensor = torch.from_dlpack(batch)
    val = tensor[0, 0, 0, 0].item()

    del batch  # drop python ref to PyBatch
    # tensor should still be valid — DLPack capsule holds shared_ptr
    assert tensor[0, 0, 0, 0].item() == val


def test_batch_data_ptr():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(4) \
        .build()

    batch = next(iter(pipeline))
    assert batch.data_ptr > 0  # nonzero device pointer


def test_batch_shape():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(4) \
        .build()

    batch = next(iter(pipeline))
    assert batch.shape == (batch.count, 3, 320, 320)


def test_no_len():
    """Pipeline deliberately does not implement __len__"""
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .batch(1) \
        .build()

    with pytest.raises(TypeError):
        len(pipeline)
