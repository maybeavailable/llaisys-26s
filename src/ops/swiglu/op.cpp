#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (gate->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out, gate, up);
    }
    llaisys::core::context().setDevice(gate->deviceType(), gate->deviceId());
    switch (gate->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out, gate, up);
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
