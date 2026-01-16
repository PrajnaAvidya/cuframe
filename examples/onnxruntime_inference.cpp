// cuframe + ONNX Runtime (CUDA EP) integration example.
//
// demonstrates the zero-copy handoff: cuframe produces NCHW float32 tensors
// on GPU, ONNX Runtime consumes them directly via device pointer using
// CUDA-backed Ort::Value bindings. no cudaMemcpy between preprocessing
// and inference.
//
// this example compiles against cuframe only. ONNX Runtime API calls are
// shown as pseudocode in comments — replace them with real ORT calls when
// linking against onnxruntime.
//
// build: cmake --preset default && cmake --build build
// run:   ./build/examples/onnxruntime_inference <video_file>

#include "cuframe/pipeline.h"
#include <cstdio>
#include <vector>

// --------------------------------------------------------------------------
// ONNX Runtime setup (pseudocode — replace with real ORT API calls)
// --------------------------------------------------------------------------
//
// #include "onnxruntime_cxx_api.h"
//
// Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cuframe");
//
// // enable CUDA execution provider
// Ort::SessionOptions session_opts;
// OrtCUDAProviderOptions cuda_opts{};
// cuda_opts.device_id = 0;
// session_opts.AppendExecutionProvider_CUDA(cuda_opts);
//
// Ort::Session session(env, "model.onnx", session_opts);
//
// // query input/output names
// auto input_name = session.GetInputNameAllocated(0, allocator);
// auto output_name = session.GetOutputNameAllocated(0, allocator);

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    // configure pipeline to match model input requirements
    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    int total = 0;
    while (auto batch = pipeline.next()) {
        auto& b = **batch;
        float* input_ptr = b.data();
        int count = b.count();

        // -------------------------------------------------------------------
        // ONNX Runtime inference (pseudocode)
        // -------------------------------------------------------------------
        //
        // the key insight: input_ptr is already a device pointer to a
        // contiguous NCHW float32 tensor. create an Ort::Value that wraps
        // this pointer directly — no copies.
        //
        // std::vector<int64_t> input_shape = {count, 3, 640, 640};
        //
        // auto mem_info = Ort::MemoryInfo::CreateCpu(
        //     OrtArenaAllocator, OrtMemTypeDefault);
        // // for GPU tensors, use CUDA memory info:
        // auto cuda_mem_info = Ort::MemoryInfo(
        //     "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0,
        //     OrtMemTypeDefault);
        //
        // auto input_tensor = Ort::Value::CreateTensor<float>(
        //     cuda_mem_info, input_ptr,
        //     count * 3 * 640 * 640,
        //     input_shape.data(), input_shape.size());
        //
        // const char* input_names[] = {input_name.get()};
        // const char* output_names[] = {output_name.get()};
        //
        // auto outputs = session.Run(
        //     Ort::RunOptions{nullptr},
        //     input_names, &input_tensor, 1,
        //     output_names, 1);
        //
        // // process output tensors...
        // -------------------------------------------------------------------

        printf("batch: %d frames → inference (device ptr %p)\n",
               count, static_cast<void*>(input_ptr));
        total += count;
    }

    printf("processed %d frames\n", total);
    return 0;
}
