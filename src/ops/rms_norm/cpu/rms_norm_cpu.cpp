#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace {

template <typename T>
void rms_norm_kernel(llaisys::tensor_t out,
                     llaisys::tensor_t in,
                     llaisys::tensor_t weight,
                     float eps) {
    const size_t rows = in->shape()[0];
    const size_t features = in->shape()[1];
    const size_t element_size = in->elementSize();
    const auto *input_data = in->data();
    const auto *weight_data = weight->data();
    auto *output_data = out->data();
    const auto &input_strides = in->strides();
    const auto &output_strides = out->strides();
    const auto weight_stride = weight->strides()[0];

    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (size_t column = 0; column < features; ++column) {
            const ptrdiff_t input_offset =
                (static_cast<ptrdiff_t>(row) * input_strides[0] +
                 static_cast<ptrdiff_t>(column) * input_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const auto value = reinterpret_cast<const T *>(input_data + input_offset);
            const float x = llaisys::utils::cast<float>(*value);
            square_sum += x * x;
        }

        const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(features) + eps);
        for (size_t column = 0; column < features; ++column) {
            const ptrdiff_t input_offset =
                (static_cast<ptrdiff_t>(row) * input_strides[0] +
                 static_cast<ptrdiff_t>(column) * input_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const ptrdiff_t output_offset =
                (static_cast<ptrdiff_t>(row) * output_strides[0] +
                 static_cast<ptrdiff_t>(column) * output_strides[1]) *
                static_cast<ptrdiff_t>(element_size);
            const auto x = reinterpret_cast<const T *>(input_data + input_offset);
            const auto w = reinterpret_cast<const T *>(
                weight_data + static_cast<ptrdiff_t>(column) * weight_stride *
                                  static_cast<ptrdiff_t>(element_size));
            auto y = reinterpret_cast<T *>(output_data + output_offset);
            *y = llaisys::utils::cast<T>(llaisys::utils::cast<float>(*x) *
                                         llaisys::utils::cast<float>(*w) * inverse_rms);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    if (in->ndim() != 2 || out->ndim() != 2 || weight->ndim() != 1) {
        throw std::invalid_argument("rms_norm expects input/output [rows, features] and weight [features]");
    }
    if (in->shape() != out->shape() || weight->shape()[0] != in->shape()[1]) {
        throw std::invalid_argument("rms_norm tensor shapes are incompatible");
    }
    if (in->dtype() != out->dtype() || in->dtype() != weight->dtype()) {
        throw std::invalid_argument("rms_norm tensors must have the same dtype");
    }
    if (in->shape()[1] == 0 || eps < 0.0f) {
        throw std::invalid_argument("rms_norm requires non-empty features and non-negative eps");
    }
    if (in->deviceType() != out->deviceType() || in->deviceId() != out->deviceId() ||
        in->deviceType() != weight->deviceType() || in->deviceId() != weight->deviceId()) {
        throw std::invalid_argument("rms_norm tensors must be on the same device");
    }

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_kernel<float>(out, in, weight, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_kernel<llaisys::fp16_t>(out, in, weight, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_kernel<llaisys::bf16_t>(out, in, weight, eps);
    default:
        throw std::invalid_argument("rms_norm supports only f32, f16, and bf16");
    }
}
}
