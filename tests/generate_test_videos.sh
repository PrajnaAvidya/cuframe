#!/bin/bash
# generate test videos for multi-codec testing
# run from project root: bash tests/generate_test_videos.sh
set -e
DIR="tests/data"
mkdir -p "$DIR"

# H.264
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libx264 -pix_fmt yuv420p "$DIR/test_h264.mp4"

# H.265 / HEVC 8-bit in mp4
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libx265 -pix_fmt yuv420p "$DIR/test_hevc.mp4"

# H.265 / HEVC 8-bit in mkv (Matroska container)
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libx265 -pix_fmt yuv420p "$DIR/test_hevc.mkv"

# VP9 8-bit in webm
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libvpx-vp9 -pix_fmt yuv420p "$DIR/test_vp9.webm"

# AV1 8-bit in mp4 (libsvtav1 is much faster than libaom)
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libsvtav1 -pix_fmt yuv420p "$DIR/test_av1.mp4"

# H.265 10-bit in mp4 (simulates iPhone HDR content)
ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=30 \
    -c:v libx265 -pix_fmt yuv420p10le "$DIR/test_hevc_10bit.mp4"

echo "done: $DIR/"
