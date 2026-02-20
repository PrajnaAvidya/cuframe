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


def test_seek_basic():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(8) \
        .build()

    pipeline.seek(0.0)
    total = 0
    for batch in pipeline:
        total += batch.count
    assert total == 90


def test_seek_middle():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(8) \
        .build()

    pipeline.seek(1.5)
    batch = pipeline.next()
    assert batch is not None


def test_seek_past_end():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(8) \
        .build()

    pipeline.seek(10.0)
    total = 0
    for batch in pipeline:
        total += batch.count
    assert total < 10


def test_seek_after_eos():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(8) \
        .build()

    # exhaust — use next() to avoid holding batch refs via loop variable
    while pipeline.next() is not None:
        pass
    assert pipeline.next() is None

    # seek back
    pipeline.seek(0.0)
    total = 0
    for batch in pipeline:
        total += batch.count
    assert total == 90


def test_seek_resets_iterator():
    """seek mid-iteration, for loop works from new position"""
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(4) \
        .build()

    # consume a few — discard refs so pool isn't exhausted after seek
    for i in range(3):
        pipeline.next()

    pipeline.seek(0.0)
    total = 0
    for batch in pipeline:
        total += batch.count
    assert total == 90


def test_temporal_stride():
    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(320, 320) \
        .batch(4) \
        .temporal_stride(2) \
        .build()

    total = 0
    for batch in pipeline:
        total += batch.count
    # 90 frames, stride=2 → 45
    assert total == 45


def test_temporal_stride_with_dlpack():
    torch = pytest.importorskip("torch")

    pipeline = cuframe.Pipeline.builder() \
        .input(VIDEO) \
        .resize(224, 224) \
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
        .batch(4) \
        .temporal_stride(2) \
        .build()

    batch = next(iter(pipeline))
    tensor = torch.from_dlpack(batch)
    assert tensor.shape == (batch.count, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cuda"


def test_threaded_pipelines():
    """GIL release in next() allows concurrent pipelines on separate threads"""
    import threading

    results = [0, 0]

    def run_pipeline(idx):
        p = cuframe.Pipeline.builder() \
            .input(VIDEO) \
            .resize(320, 320) \
            .batch(8) \
            .build()
        for batch in p:
            results[idx] += batch.count

    t0 = threading.Thread(target=run_pipeline, args=(0,))
    t1 = threading.Thread(target=run_pipeline, args=(1,))
    t0.start()
    t1.start()
    t0.join(timeout=30)
    t1.join(timeout=30)
    assert not t0.is_alive()
    assert not t1.is_alive()
    assert results[0] == 90
    assert results[1] == 90
