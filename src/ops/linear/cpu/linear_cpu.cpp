#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <stdexcept>

namespace {

template <typename T>
void linear_kernel(llaisys::tensor_t out,
                   llaisys::tensor_t in,
                   llaisys::tensor_t weight,
                   llaisys::tensor_t bias) {
    const size_t rows = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    const size_t element_size = in->elementSize();
    const auto *in_data = in->data();
    const auto *weight_data = weight->data();
    auto *out_data = out->data();
    const auto &in_strides = in->strides();
    const auto &weight_strides = weight->strides();
    const auto &out_strides = out->strides();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t output_col = 0; output_col < out_features; ++output_col) {
            float value = 0.0f;
            for (size_t input_col = 0; input_col < in_features; ++input_col) {
                const auto *x = reinterpret_cast<const T *>(
                    in_data + (static_cast<ptrdiff_t>(row) * in_strides[0] +
                               static_cast<ptrdiff_t>(input_col) * in_strides[1]) *
                                  static_cast<ptrdiff_t>(element_size));
                const auto *w = reinterpret_cast<const T *>(
                    weight_data + (static_cast<ptrdiff_t>(output_col) * weight_strides[0] +
                                   static_cast<ptrdiff_t>(input_col) * weight_strides[1]) *
                                      static_cast<ptrdiff_t>(element_size));
                value += llaisys::utils::cast<float>(*x) * llaisys::utils::cast<float>(*w);
            }
            if (bias) {
                const auto *b = reinterpret_cast<const T *>(bias->data()) +
                                output_col * bias->strides()[0];
                value += llaisys::utils::cast<float>(*b);
            }
            auto *result = reinterpret_cast<T *>(
                out_data + (static_cast<ptrdiff_t>(row) * out_strides[0] +
                            static_cast<ptrdiff_t>(output_col) * out_strides[1]) *
                               static_cast<ptrdiff_t>(element_size));
            *result = llaisys::utils::cast<T>(value);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (in->ndim() != 2 || weight->ndim() != 2 || out->ndim() != 2) {
        throw std::invalid_argument("linear expects 2D input, weight, and output tensors");
    }
    if (in->shape()[1] != weight->shape()[1] ||
        out->shape()[0] != in->shape()[0] ||
        out->shape()[1] != weight->shape()[0]) {
        throw std::invalid_argument("linear tensor shapes are incompatible");
    }
    if (in->dtype() != weight->dtype() || in->dtype() != out->dtype() ||
        (bias && bias->dtype() != in->dtype())) {
        throw std::invalid_argument("linear tensors must have the same dtype");
    }
    if (bias && (bias->ndim() != 1 || bias->shape()[0] != weight->shape()[0])) {
        throw std::invalid_argument("linear bias shape must be [weight.size(0)]");
    }
    if (in->deviceType() != weight->deviceType() ||
        in->deviceId() != weight->deviceId() ||
        out->deviceType() != in->deviceType() || out->deviceId() != in->deviceId() ||
        (bias && (bias->deviceType() != in->deviceType() ||
                  bias->deviceId() != in->deviceId()))) {
        throw std::invalid_argument("linear tensors must be on the same device");
    }

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_kernel<float>(out, in, weight, bias);
    case LLAISYS_DTYPE_F16:
        return linear_kernel<llaisys::fp16_t>(out, in, weight, bias);
    case LLAISYS_DTYPE_BF16:
        return linear_kernel<llaisys::bf16_t>(out, in, weight, bias);
    default:
        throw std::invalid_argument("linear supports only f32, f16, and bf16");
    }
}
}
