#include "argmax_cpu.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

float read_as_float(const std::byte *ptr, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return *reinterpret_cast<const float *>(ptr);
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::_f16_to_f32(
            *reinterpret_cast<const llaisys::fp16_t *>(ptr));
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::_bf16_to_f32(
            *reinterpret_cast<const llaisys::bf16_t *>(ptr));
    default:
        throw std::invalid_argument("argmax supports only f32, f16, and bf16 inputs");
    }
}

} // namespace

void argmax_(llaisys::tensor_t max_idx,
             llaisys::tensor_t max_val,
             llaisys::tensor_t vals) {
    if (vals->ndim() != 1) {
        throw std::invalid_argument("argmax expects a 1D input tensor");
    }
    if (vals->numel() == 0) {
        throw std::invalid_argument("argmax cannot reduce an empty tensor");
    }
    if (max_idx->numel() != 1 || max_val->numel() != 1) {
        throw std::invalid_argument("argmax outputs must each contain one element");
    }
    if (max_idx->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("argmax max_idx must have i64 dtype");
    }
    if (max_val->dtype() != vals->dtype()) {
        throw std::invalid_argument("argmax max_val dtype must match input dtype");
    }
    if (max_idx->deviceType() != vals->deviceType() ||
        max_idx->deviceId() != vals->deviceId() ||
        max_val->deviceType() != vals->deviceType() ||
        max_val->deviceId() != vals->deviceId()) {
        throw std::invalid_argument("argmax inputs and outputs must be on the same device");
    }

    const size_t element_size = vals->elementSize();
    const ptrdiff_t input_stride = vals->strides()[0];
    const std::byte *input = vals->data();
    const std::byte *max_element = input;
    float max_value = read_as_float(max_element, vals->dtype());
    int64_t max_index = 0;

    for (size_t i = 1; i < vals->numel(); ++i) {
        const std::byte *element = input +
            static_cast<ptrdiff_t>(i) * input_stride * static_cast<ptrdiff_t>(element_size);
        const float value = read_as_float(element, vals->dtype());

        // Match torch.max: NaN wins, and ties retain the first occurrence.
        if ((!std::isnan(max_value) && std::isnan(value)) ||
            (!std::isnan(value) && value > max_value)) {
            max_value = value;
            max_index = static_cast<int64_t>(i);
            max_element = element;
        }
    }

    std::memcpy(max_val->data(), max_element, element_size);
    *reinterpret_cast<int64_t *>(max_idx->data()) = max_index;
}

namespace llaisys::ops::cpu {

void argmax(llaisys::tensor_t max_idx,
            llaisys::tensor_t max_val,
            llaisys::tensor_t vals) {
    argmax_(max_idx, max_val, vals);
}

} // namespace llaisys::ops::cpu
