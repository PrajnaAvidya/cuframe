#!/bin/bash
# build release and run benchmarks
# usage: ./tools/bench.sh [video_file]
#   default video: /tmp/bench_1080p.mp4

set -e

VIDEO="${1:-/tmp/bench_1080p.mp4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

if [[ ! -f "$VIDEO" ]]; then
    echo "generating test video: $VIDEO"
    ffmpeg -f lavfi -i testsrc2=duration=10:size=1920x1080:rate=30 \
        -c:v libx264 -pix_fmt yuv420p "$VIDEO" -y 2>/dev/null
fi

echo "=== configuring (Release) ==="
cmake --preset default -DCMAKE_BUILD_TYPE=Release -S "$ROOT" -B "$BUILD" 2>&1 | tail -3

echo ""
echo "=== building ==="
cmake --build "$BUILD" -j 2>&1 | tail -3

echo ""
echo "=== bench_pipeline ==="
"$BUILD/benchmarks/bench_pipeline" "$VIDEO"

echo ""
echo "=== bench_preprocess ==="
"$BUILD/benchmarks/bench_preprocess" "$VIDEO"
