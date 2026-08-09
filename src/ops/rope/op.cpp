#include "op.hpp"
#include "cpu/rope_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out, in, pos_ids, theta);
    }
    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out, in, pos_ids, theta);
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
