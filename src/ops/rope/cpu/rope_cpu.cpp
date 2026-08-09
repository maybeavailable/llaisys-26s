#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <stdexcept>

namespace {

template <typename T>
void rope_kernel(llaisys::tensor_t out,
                 llaisys::tensor_t in,
                 llaisys::tensor_t pos_ids,
                 float theta) {
    const size_t sequence_length = in->shape()[0];
    const size_t heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];
    const size_t half_dim = head_dim / 2;
    const size_t element_size = in->elementSize();
    const auto &input_strides = in->strides();
    const auto &output_strides = out->strides();
    const auto *input_data = in->data();
    auto *output_data = out->data();
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids->data());

    for (size_t sequence = 0; sequence < sequence_length; ++sequence) {
        const float position = static_cast<float>(positions[
            static_cast<ptrdiff_t>(sequence) * pos_ids->strides()[0]]);
        for (size_t head = 0; head < heads; ++head) {
            for (size_t j = 0; j < half_dim; ++j) {
                const float exponent =
                    2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
                const float angle = position / std::pow(theta, exponent);
                const float sine = std::sin(angle);
                const float cosine = std::cos(angle);
                const ptrdiff_t base_input =
                    static_cast<ptrdiff_t>(sequence) * input_strides[0] +
                    static_cast<ptrdiff_t>(head) * input_strides[1];
                const ptrdiff_t base_output =
                    static_cast<ptrdiff_t>(sequence) * output_strides[0] +
                    static_cast<ptrdiff_t>(head) * output_strides[1];
                const auto *a = reinterpret_cast<const T *>(
                    input_data + (base_input + static_cast<ptrdiff_t>(j) * input_strides[2]) *
                                     static_cast<ptrdiff_t>(element_size));
                const auto *b = reinterpret_cast<const T *>(
                    input_data + (base_input + static_cast<ptrdiff_t>(j + half_dim) * input_strides[2]) *
                                     static_cast<ptrdiff_t>(element_size));
                auto *a_out = reinterpret_cast<T *>(
                    output_data + (base_output + static_cast<ptrdiff_t>(j) * output_strides[2]) *
                                      static_cast<ptrdiff_t>(element_size));
                auto *b_out = reinterpret_cast<T *>(
                    output_data + (base_output + static_cast<ptrdiff_t>(j + half_dim) * output_strides[2]) *
                                      static_cast<ptrdiff_t>(element_size));
                const float a_value = llaisys::utils::cast<float>(*a);
                const float b_value = llaisys::utils::cast<float>(*b);
                *a_out = llaisys::utils::cast<T>(a_value * cosine - b_value * sine);
                *b_out = llaisys::utils::cast<T>(b_value * cosine + a_value * sine);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    if (in->ndim() != 3 || out->ndim() != 3 || pos_ids->ndim() != 1) {
        throw std::invalid_argument("rope expects input/output [seq, heads, dim] and pos_ids [seq]");
    }
    if (in->shape() != out->shape() || in->shape()[0] != pos_ids->shape()[0]) {
        throw std::invalid_argument("rope tensor shapes are incompatible");
    }
    if (in->shape()[2] == 0 || in->shape()[2] % 2 != 0 || theta <= 0.0f) {
        throw std::invalid_argument("rope requires a positive even head dimension and theta");
    }
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("rope pos_ids must have i64 dtype");
    }
    if (in->dtype() != out->dtype()) {
        throw std::invalid_argument("rope input and output must have the same dtype");
    }
    if (in->deviceType() != out->deviceType() || in->deviceId() != out->deviceId() ||
        in->deviceType() != pos_ids->deviceType() || in->deviceId() != pos_ids->deviceId()) {
        throw std::invalid_argument("rope tensors must be on the same device");
    }

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_kernel<float>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_F16:
        return rope_kernel<llaisys::fp16_t>(out, in, pos_ids, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_kernel<llaisys::bf16_t>(out, in, pos_ids, theta);
    default:
        throw std::invalid_argument("rope supports only f32, f16, and bf16");
    }
}
}
