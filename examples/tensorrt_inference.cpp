// cuframe + TensorRT integration example.
//
// demonstrates the zero-copy handoff: cuframe produces NCHW float32 tensors
// on GPU, TensorRT consumes them directly via device pointer. no cudaMemcpy
// between preprocessing and inference.
//
// this example compiles against cuframe only. TensorRT API calls are shown
// as pseudocode in comments — replace them with real TRT calls when linking
// against nvinfer.
//
// build: cmake --preset default && cmake --build build
// run:   ./build/examples/tensorrt_inference <video_file>

#include <cuframe/cuframe.h>
#include <cstdio>
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// TensorRT setup (pseudocode — replace with real TRT API calls)
// --------------------------------------------------------------------------
//
// #include "NvInfer.h"
//
// // load a serialized TensorRT engine
// auto runtime = nvinfer1::createInferRuntime(logger);
// auto engine = runtime->deserializeCudaEngine(engine_data, engine_size);
// auto context = engine->createExecutionContext();
//
// // find input/output tensor names and shapes
// const char* input_name = "images";       // model-specific
// const char* output_name = "output0";     // model-specific
//
// // allocate output buffer on GPU
// size_t output_size = /* depends on model */ 0;
// void* d_output = nullptr;
// cudaMalloc(&d_output, output_size);

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <video_file>\n", argv[0]);
        return 1;
    }

    // configure pipeline to match model input requirements
    // example: YOLOv8 expects 640x640 letterboxed, ImageNet-normalized
    auto pipeline = cuframe::Pipeline::builder()
        .input(argv[1])
        .resize(640, 640, cuframe::ResizeMode::LETTERBOX)
        .normalize({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f})
        .batch(8)
        .build();

    printf("source: %dx%d, %.2f fps, %lld frames\n",
           pipeline.source_width(), pipeline.source_height(),
           pipeline.fps(), (long long)pipeline.frame_count());

    cudaStream_t infer_stream;
    cudaStreamCreate(&infer_stream);

    int total = 0;
    while (auto batch = pipeline.next()) {
        auto& b = **batch;
        float* input_ptr = b.data();
        int count = b.count();

        // -------------------------------------------------------------------
        // TensorRT inference (pseudocode)
        // -------------------------------------------------------------------
        //
        // the key insight: input_ptr is already a device pointer to a
        // contiguous NCHW float32 tensor. no copies needed — just bind it
        // as the input tensor and run inference.
        //
        // context->setTensorAddress(input_name, input_ptr);
        // context->setTensorAddress(output_name, d_output);
        // context->setInputShape(input_name, nvinfer1::Dims4{count, 3, 640, 640});
        // context->enqueueV3(infer_stream);
        // cudaStreamSynchronize(infer_stream);
        //
        // // process detections from d_output...
        // -------------------------------------------------------------------

        printf("batch: %d frames → inference (device ptr %p)\n",
               count, static_cast<void*>(input_ptr));
        total += count;
    }

    printf("processed %d frames\n", total);

    // cleanup
    cudaStreamDestroy(infer_stream);

    // TensorRT cleanup:
    // cudaFree(d_output);
    // delete context;
    // delete engine;
    // delete runtime;

    return 0;
}
