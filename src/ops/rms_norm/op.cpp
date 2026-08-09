#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out, in, weight, eps);
    }
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out, in, weight, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
