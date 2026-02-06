#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <cuframe/pipeline.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// DLPack managed tensor — ABI-compatible with upstream dlpack/dlpack.h.
// nanobind provides the dltensor struct but not the managed wrapper.
struct DLManagedTensor {
    nb::dlpack::dltensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

constexpr int kDLCUDA = 2;

// thin wrapper: carries shared_ptr (for DLPack lifetime) + device_id.
// pipeline_ref keeps the pipeline alive — its BatchPool owns the underlying
// memory, so we must prevent the pool from being destroyed while any batch exists.
// member order matters: C++ destroys in reverse order, so pipeline_ref (declared
// first) is destroyed last, ensuring the pool outlives the shared_ptr.
struct PyBatch {
    nb::object pipeline_ref;
    std::shared_ptr<cuframe::GpuFrameBatch> ptr;
    int device_id;
};

// PyCapsule destructor for DLPack protocol. when torch consumes the capsule,
// it renames it from "dltensor" to "used_dltensor" — so we only call the
// managed deleter if nobody consumed it.
void dlpack_capsule_destructor(PyObject* capsule) {
    if (strcmp(PyCapsule_GetName(capsule), "used_dltensor") == 0)
        return;  // consumer took ownership, will call deleter itself
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed && managed->deleter)
        managed->deleter(managed);
}

// DLPack context — holds both the batch shared_ptr and a pipeline reference.
// uses raw PyObject* because torch may call the deleter without the GIL,
// and nb::object's destructor requires GIL for refcount operations.
struct DLPackContext {
    PyObject* pipeline_ref;  // prevents pipeline/pool destruction
    std::shared_ptr<cuframe::GpuFrameBatch> batch;  // prevents pool return
};

nb::object batch_dlpack(PyBatch& self, int64_t stream) {
    auto& batch = *self.ptr;

    PyObject* pipeline_py = self.pipeline_ref.ptr();
    Py_INCREF(pipeline_py);
    auto* ctx = new DLPackContext{pipeline_py, self.ptr};

    auto* managed = new DLManagedTensor;
    managed->dl_tensor.data = static_cast<void*>(batch.data());
    managed->dl_tensor.device = {kDLCUDA, self.device_id};
    managed->dl_tensor.ndim = 4;
    managed->dl_tensor.dtype = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    managed->dl_tensor.shape = new int64_t[4]{
        batch.count(), batch.channels(), batch.height(), batch.width()};
    managed->dl_tensor.strides = nullptr;  // contiguous NCHW
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = ctx;
    managed->deleter = [](DLManagedTensor* t) {
        auto* ctx = static_cast<DLPackContext*>(t->manager_ctx);
        // drop batch first — shared_ptr deleter accesses pool via captured pointer
        ctx->batch.reset();
        // release pipeline ref (acquires GIL since torch may call without it)
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_DECREF(ctx->pipeline_ref);
        PyGILState_Release(gstate);
        delete ctx;
        delete[] t->dl_tensor.shape;
        delete t;
    };

    PyObject* capsule = PyCapsule_New(managed, "dltensor", dlpack_capsule_destructor);
    return nb::steal(capsule);
}

nb::tuple batch_dlpack_device(PyBatch& self) {
    return nb::make_tuple(kDLCUDA, self.device_id);
}

} // anonymous namespace

NB_MODULE(_cuframe, m) {
    m.doc() = "gpu video preprocessing — zero-copy decode to inference-ready tensors";

    // --- ResizeMode enum ---
    nb::enum_<cuframe::ResizeMode>(m, "ResizeMode")
        .value("STRETCH", cuframe::ResizeMode::STRETCH)
        .value("LETTERBOX", cuframe::ResizeMode::LETTERBOX);

    // --- GpuFrameBatch (via PyBatch wrapper) ---
    nb::class_<PyBatch>(m, "GpuFrameBatch")
        .def_prop_ro("batch_size", [](PyBatch& s) { return s.ptr->batch_size(); })
        .def_prop_ro("count", [](PyBatch& s) { return s.ptr->count(); })
        .def_prop_ro("channels", [](PyBatch& s) { return s.ptr->channels(); })
        .def_prop_ro("height", [](PyBatch& s) { return s.ptr->height(); })
        .def_prop_ro("width", [](PyBatch& s) { return s.ptr->width(); })
        .def("__dlpack__", &batch_dlpack, "stream"_a = 0)
        .def("__dlpack_device__", &batch_dlpack_device)
        .def("__repr__", [](PyBatch& s) {
            auto& b = *s.ptr;
            return "GpuFrameBatch(count=" + std::to_string(b.count()) +
                   ", channels=" + std::to_string(b.channels()) +
                   ", height=" + std::to_string(b.height()) +
                   ", width=" + std::to_string(b.width()) + ")";
        });

    // --- PipelineBuilder ---
    nb::class_<cuframe::PipelineBuilder>(m, "PipelineBuilder")
        .def("input", &cuframe::PipelineBuilder::input, nb::rv_policy::reference)
        .def("resize", &cuframe::PipelineBuilder::resize,
             "width"_a, "height"_a,
             "mode"_a = cuframe::ResizeMode::LETTERBOX,
             "pad_value"_a = 114.0f,
             nb::rv_policy::reference)
        .def("normalize", &cuframe::PipelineBuilder::normalize,
             "mean"_a, "std"_a,
             nb::rv_policy::reference)
        .def("center_crop", &cuframe::PipelineBuilder::center_crop,
             "width"_a, "height"_a,
             nb::rv_policy::reference)
        .def("batch", &cuframe::PipelineBuilder::batch,
             "size"_a,
             nb::rv_policy::reference)
        .def("pool_size", &cuframe::PipelineBuilder::pool_size,
             "n"_a,
             nb::rv_policy::reference)
        .def("channel_order_bgr", &cuframe::PipelineBuilder::channel_order_bgr,
             "bgr"_a = true,
             nb::rv_policy::reference)
        .def("retain_decoded", &cuframe::PipelineBuilder::retain_decoded,
             "retain"_a = true,
             nb::rv_policy::reference)
        .def("device", &cuframe::PipelineBuilder::device,
             "gpu_id"_a,
             nb::rv_policy::reference)
        .def("build", [](cuframe::PipelineBuilder& self) {
            return self.build();
        });

    // --- Pipeline ---
    nb::class_<cuframe::Pipeline>(m, "Pipeline")
        .def_static("builder", &cuframe::Pipeline::builder)
        .def("next", [](nb::object self_obj) -> nb::object {
            auto& self = nb::cast<cuframe::Pipeline&>(self_obj);
            auto batch = self.next();
            if (!batch) return nb::none();
            return nb::cast(PyBatch{self_obj, *batch, self.config().device_id});
        })
        .def("__iter__", [](cuframe::Pipeline& self) -> cuframe::Pipeline& {
            return self;
        }, nb::rv_policy::reference)
        .def("__next__", [](nb::object self_obj) -> PyBatch {
            auto& self = nb::cast<cuframe::Pipeline&>(self_obj);
            auto batch = self.next();
            if (!batch) throw nb::stop_iteration();
            return PyBatch{self_obj, *batch, self.config().device_id};
        })
        .def_prop_ro("source_width", &cuframe::Pipeline::source_width)
        .def_prop_ro("source_height", &cuframe::Pipeline::source_height)
        .def_prop_ro("fps", &cuframe::Pipeline::fps)
        .def_prop_ro("frame_count", &cuframe::Pipeline::frame_count);
}
