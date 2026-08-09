#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace {

template <typename T>
void swiglu_kernel(llaisys::tensor_t out,
                   llaisys::tensor_t gate,
                   llaisys::tensor_t up) {
    const size_t rows = gate->shape()[0];
    const size_t columns = gate->shape()[1];
    const size_t element_size = gate->elementSize();
    const auto &gate_strides = gate->strides();
    const auto &up_strides = up->strides();
    const auto &out_strides = out->strides();
    const auto *gate_data = gate->data();
    const auto *up_data = up->data();
    auto *out_data = out->data();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const auto gate_offset =
                (static_cast<ptrdiff_t>(row) * gate_strides[0] +
                 static_cast<ptrdiff_t>(column) * gate_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const auto up_offset =
                (static_cast<ptrdiff_t>(row) * up_strides[0] +
                 static_cast<ptrdiff_t>(column) * up_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const auto out_offset =
                (static_cast<ptrdiff_t>(row) * out_strides[0] +
                 static_cast<ptrdiff_t>(column) * out_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const float gate_value = llaisys::utils::cast<float>(
                *reinterpret_cast<const T *>(gate_data + gate_offset));
            const float up_value = llaisys::utils::cast<float>(
                *reinterpret_cast<const T *>(up_data + up_offset));
            const float sigmoid = gate_value >= 0.0f
                ? 1.0f / (1.0f + std::exp(-gate_value))
                : std::exp(gate_value) / (1.0f + std::exp(gate_value));
            *reinterpret_cast<T *>(out_data + out_offset) =
                llaisys::utils::cast<T>(up_value * gate_value * sigmoid);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    if (gate->ndim() != 2 || up->ndim() != 2 || out->ndim() != 2 ||
        gate->shape() != up->shape() || gate->shape() != out->shape()) {
        throw std::invalid_argument("swiglu expects gate, up, and out tensors with the same 2D shape");
    }
    if (gate->dtype() != up->dtype() || gate->dtype() != out->dtype()) {
        throw std::invalid_argument("swiglu tensors must have the same dtype");
    }
    if (gate->deviceType() != up->deviceType() || gate->deviceId() != up->deviceId() ||
        gate->deviceType() != out->deviceType() || gate->deviceId() != out->deviceId()) {
        throw std::invalid_argument("swiglu tensors must be on the same device");
    }

    switch (gate->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_kernel<float>(out, gate, up);
    case LLAISYS_DTYPE_F16:
        return swiglu_kernel<llaisys::fp16_t>(out, gate, up);
    case LLAISYS_DTYPE_BF16:
        return swiglu_kernel<llaisys::bf16_t>(out, gate, up);
    default:
        throw std::invalid_argument("swiglu supports only f32, f16, and bf16");
    }
}
}
