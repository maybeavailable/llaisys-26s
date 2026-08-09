#include "op.hpp"
#include "cpu/linear_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out, in, weight, bias);
    }
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias);
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
