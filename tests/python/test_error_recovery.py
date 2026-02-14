import cuframe
import pytest
import os

VIDEO = "tests/data/test_h264.mp4"
CORRUPT = "tests/data/test_corrupt_py.mp4"


@pytest.fixture(autouse=True)
def corrupt_video():
    if not os.path.exists(VIDEO):
        pytest.skip("test video not found")
    # heavy corruption: keep moov intact but wipe 40% of mdat payload
    with open(VIDEO, "rb") as f:
        data = bytearray(f.read())
    start = len(data) // 4
    length = len(data) * 2 // 5
    for i in range(length):
        if start + i < len(data):
            data[start + i] = 0xFF
    with open(CORRUPT, "wb") as f:
        f.write(data)
    yield
    if os.path.exists(CORRUPT):
        os.remove(CORRUPT)


def test_error_policy_enum():
    assert cuframe.ErrorPolicy.THROW != cuframe.ErrorPolicy.SKIP


def test_default_policy_throws():
    pipeline = (cuframe.Pipeline.builder()
        .input(CORRUPT)
        .resize(640, 640)
        .batch(8)
        .build())

    with pytest.raises(Exception):
        for _ in pipeline:
            pass


def test_skip_policy_no_throw():
    pipeline = (cuframe.Pipeline.builder()
        .input(CORRUPT)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe.ErrorPolicy.SKIP)
        .build())

    total = 0
    for batch in pipeline:
        total += batch.count

    assert total > 0
    assert total < 90
    assert pipeline.error_count > 0


def test_skip_callback():
    errors = []

    pipeline = (cuframe.Pipeline.builder()
        .input(CORRUPT)
        .resize(640, 640)
        .batch(8)
        .error_policy(cuframe.ErrorPolicy.SKIP)
        .on_error(lambda info: errors.append(info.message))
        .build())

    for _ in pipeline:
        pass

    assert len(errors) > 0
    assert len(errors) == pipeline.error_count
    assert all(isinstance(msg, str) for msg in errors)


def test_skip_valid_file():
    pipeline = (cuframe.Pipeline.builder()
        .input(VIDEO)
        .resize(640, 640)
        .normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        .batch(8)
        .error_policy(cuframe.ErrorPolicy.SKIP)
        .build())

    total = 0
    for batch in pipeline:
        total += batch.count

    assert total == 90
    assert pipeline.error_count == 0


def test_error_count_zero_initially():
    pipeline = (cuframe.Pipeline.builder()
        .input(VIDEO)
        .batch(1)
        .build())

    assert pipeline.error_count == 0
