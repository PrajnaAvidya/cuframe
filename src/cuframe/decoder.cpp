#include "cuframe/decoder.h"
#include "cuframe/cuda_utils.h"

#include <stdexcept>
#include <string>

namespace cuframe {

cudaVideoCodec to_nvdec_codec(AVCodecID id) {
    switch (id) {
        case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
        case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
        case AV_CODEC_ID_VP9:  return cudaVideoCodec_VP9;
        case AV_CODEC_ID_AV1:  return cudaVideoCodec_AV1;
        default:
            throw std::runtime_error(
                "unsupported codec for nvdec: " + std::to_string(static_cast<int>(id)));
    }
}

Decoder::Decoder(const VideoInfo& info, int pool_count)
    : pool_count_(pool_count) {
    cuInit(0);
    // force runtime context creation so cuCtxGetCurrent works
    cudaFree(0);
    CUFRAME_CU_CHECK(cuCtxGetCurrent(&cu_ctx_));
    CUFRAME_CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));

    CUVIDPARSERPARAMS parser_params = {};
    parser_params.CodecType = to_nvdec_codec(info.codec_id);
    parser_params.ulMaxNumDecodeSurfaces = 1;
    parser_params.ulMaxDisplayDelay = 1;
    parser_params.pUserData = this;
    parser_params.pfnSequenceCallback = handle_sequence;
    parser_params.pfnDecodePicture = handle_decode;
    parser_params.pfnDisplayPicture = handle_display;

    CUFRAME_CU_CHECK(cuvidCreateVideoParser(&parser_, &parser_params));
}

Decoder::~Decoder() {
    if (parser_) cuvidDestroyVideoParser(parser_);
    if (stream_) cuStreamDestroy(stream_);
    if (decoder_) {
        cuCtxPushCurrent(cu_ctx_);
        cuvidDestroyDecoder(decoder_);
        cuCtxPopCurrent(nullptr);
    }
}

void Decoder::decode(const AVPacket* packet, std::vector<DecodedFrame>& out) {
    CUVIDSOURCEDATAPACKET cupkt = {};
    cupkt.payload = packet->data;
    cupkt.payload_size = packet->size;
    cupkt.flags = CUVID_PKT_TIMESTAMP;
    cupkt.timestamp = packet->pts;

    CUFRAME_CU_CHECK(cuvidParseVideoData(parser_, &cupkt));

    // callbacks fire synchronously — pending_frames_ now populated
    for (auto& f : pending_frames_)
        out.push_back(std::move(f));
    pending_frames_.clear();
}

void Decoder::flush(std::vector<DecodedFrame>& out) {
    CUVIDSOURCEDATAPACKET cupkt = {};
    cupkt.flags = CUVID_PKT_ENDOFSTREAM;

    CUFRAME_CU_CHECK(cuvidParseVideoData(parser_, &cupkt));

    for (auto& f : pending_frames_)
        out.push_back(std::move(f));
    pending_frames_.clear();
}

int Decoder::width() const { return width_; }
int Decoder::height() const { return height_; }
CUstream Decoder::stream() const { return stream_; }

// --- callbacks ---

int Decoder::on_sequence(CUVIDEOFORMAT* fmt) {
    // resolution change mid-stream: tear down old decoder + pool
    if (decoder_) {
        pool_.reset();
        cuCtxPushCurrent(cu_ctx_);
        cuvidDestroyDecoder(decoder_);
        cuCtxPopCurrent(nullptr);
        decoder_ = nullptr;
    }

    width_ = fmt->display_area.right - fmt->display_area.left;
    height_ = fmt->display_area.bottom - fmt->display_area.top;
    surface_height_ = fmt->coded_height;

    CUVIDDECODECREATEINFO ci = {};
    ci.CodecType = fmt->codec;
    ci.ChromaFormat = fmt->chroma_format;
    ci.OutputFormat = cudaVideoSurfaceFormat_NV12;
    ci.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
    ci.DeinterlaceMode = fmt->progressive_sequence
        ? cudaVideoDeinterlaceMode_Weave
        : cudaVideoDeinterlaceMode_Adaptive;
    ci.ulNumDecodeSurfaces = fmt->min_num_decode_surfaces;
    ci.ulNumOutputSurfaces = 2;
    ci.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    ci.ulWidth = fmt->coded_width;
    ci.ulHeight = fmt->coded_height;
    ci.ulMaxWidth = fmt->coded_width;
    ci.ulMaxHeight = fmt->coded_height;
    ci.ulTargetWidth = fmt->coded_width;
    ci.ulTargetHeight = fmt->coded_height;

    cuCtxPushCurrent(cu_ctx_);
    CUFRAME_CU_CHECK(cuvidCreateDecoder(&decoder_, &ci));
    cuCtxPopCurrent(nullptr);

    return fmt->min_num_decode_surfaces;
}

int Decoder::on_decode(CUVIDPICPARAMS* pic) {
    cuCtxPushCurrent(cu_ctx_);
    CUFRAME_CU_CHECK(cuvidDecodePicture(decoder_, pic));
    cuCtxPopCurrent(nullptr);
    return 1;
}

int Decoder::on_display(CUVIDPARSERDISPINFO* disp) {
    if (!disp) return 1;  // null = EOS notification

    CUVIDPROCPARAMS map_params = {};
    map_params.progressive_frame = disp->progressive_frame;
    map_params.second_field = disp->repeat_first_field + 1;
    map_params.top_field_first = disp->top_field_first;
    map_params.unpaired_field = disp->repeat_first_field < 0;

    cuCtxPushCurrent(cu_ctx_);

    // map nvdec surface to get device pointer
    unsigned long long src_ptr = 0;
    unsigned int src_pitch = 0;
    CUFRAME_CU_CHECK(cuvidMapVideoFrame(decoder_, disp->picture_index,
                                         &src_ptr, &src_pitch, &map_params));

    // NV12: luma (height * pitch) + chroma (height/2 * pitch)
    unsigned int luma_height = height_;
    unsigned int chroma_height = (height_ + 1) / 2;
    size_t frame_size = static_cast<size_t>(src_pitch) * (luma_height + chroma_height);

    // lazy pool creation — need actual src_pitch from first mapped frame
    if (!pool_) {
        pool_ = std::make_unique<FramePool>(frame_size, pool_count_);
    }

    auto buf = pool_->acquire();
    if (!buf) {
        CUFRAME_CU_CHECK(cuvidUnmapVideoFrame(decoder_, src_ptr));
        cuCtxPopCurrent(nullptr);
        throw std::runtime_error("frame pool exhausted");
    }

    // copy luma plane
    CUDA_MEMCPY2D cp = {};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = src_ptr;
    cp.srcPitch = src_pitch;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = reinterpret_cast<CUdeviceptr>(buf->data());
    cp.dstPitch = src_pitch;
    cp.WidthInBytes = src_pitch;  // copy full pitched rows (no uninitialized gaps)
    cp.Height = luma_height;
    CUFRAME_CU_CHECK(cuMemcpy2DAsync(&cp, stream_));

    // copy chroma plane (interleaved UV, offset by aligned surface height)
    cp.srcDevice = src_ptr + src_pitch * ((surface_height_ + 1) & ~1u);
    cp.dstDevice = reinterpret_cast<CUdeviceptr>(buf->data()) + src_pitch * luma_height;
    cp.Height = chroma_height;
    CUFRAME_CU_CHECK(cuMemcpy2DAsync(&cp, stream_));

    CUFRAME_CU_CHECK(cuStreamSynchronize(stream_));
    CUFRAME_CU_CHECK(cuvidUnmapVideoFrame(decoder_, src_ptr));

    cuCtxPopCurrent(nullptr);

    pending_frames_.push_back(DecodedFrame{
        std::move(buf), width_, height_, src_pitch, disp->timestamp, stream_
    });

    return 1;
}

// --- static trampolines ---

int CUDAAPI Decoder::handle_sequence(void* ud, CUVIDEOFORMAT* fmt) {
    return static_cast<Decoder*>(ud)->on_sequence(fmt);
}

int CUDAAPI Decoder::handle_decode(void* ud, CUVIDPICPARAMS* pic) {
    return static_cast<Decoder*>(ud)->on_decode(pic);
}

int CUDAAPI Decoder::handle_display(void* ud, CUVIDPARSERDISPINFO* disp) {
    return static_cast<Decoder*>(ud)->on_display(disp);
}

} // namespace cuframe
