#include "op.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "../../utils/check.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val, q, k, v, scale);
    }
    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());
    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val, q, k, v, scale);
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
