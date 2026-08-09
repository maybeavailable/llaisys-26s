#include "op.hpp"
#include "cpu/rearrange_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out, in);
    }
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out, in);
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
