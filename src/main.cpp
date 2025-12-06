#include <cuda.h>
#include <cuda_runtime.h>
#include <cuviddec.h>
#include <nvcuvid.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include <cstdio>

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("device %d: %s (sm_%d%d, %.1f GB)\n",
               i, props.name, props.major, props.minor,
               props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    printf("ffmpeg: %s\n", av_version_info());

    // init driver api and force context creation (required before cuvid calls)
    cuInit(0);
    cudaFree(0);

    // verify nvdec symbol resolves
    CUVIDDECODECAPS caps = {};
    caps.eCodecType = cudaVideoCodec_H264;
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = 0;
    CUresult res = cuvidGetDecoderCaps(&caps);
    if (res == CUDA_SUCCESS && caps.bIsSupported) {
        printf("nvdec: h264 decode supported, max %dx%d\n",
               caps.nMaxWidth, caps.nMaxHeight);
    } else {
        printf("nvdec: cuvidGetDecoderCaps returned %d\n", res);
    }

    return 0;
}
