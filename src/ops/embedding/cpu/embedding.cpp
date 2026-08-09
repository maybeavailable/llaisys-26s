#include "embedding.hpp"

#include <cstring>
#include <stdexcept>

namespace {

void embedding_(llaisys::tensor_t out,
                llaisys::tensor_t index,
                llaisys::tensor_t weight) {
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("embedding index tensor must have i64 dtype");
    }
    if (index->ndim() != 1 || weight->ndim() != 2 || out->ndim() != 2) {
        throw std::invalid_argument("embedding expects index [N], weight [V, D], and out [N, D]");
    }
    if (out->dtype() != weight->dtype()) {
        throw std::invalid_argument("embedding output dtype must match weight dtype");
    }
    if (out->shape()[0] != index->shape()[0] ||
        out->shape()[1] != weight->shape()[1]) {
        throw std::invalid_argument("embedding output shape must be [index.size(0), weight.size(1)]");
    }
    if (out->deviceType() != weight->deviceType() ||
        out->deviceId() != weight->deviceId() ||
        index->deviceType() != weight->deviceType() ||
        index->deviceId() != weight->deviceId()) {
        throw std::invalid_argument("embedding tensors must be on the same device");
    }

    const size_t element_size = weight->elementSize();
    const auto &index_strides = index->strides();
    const auto &weight_strides = weight->strides();
    const auto &out_strides = out->strides();
    const auto *index_data = reinterpret_cast<const int64_t *>(index->data());
    const std::byte *weight_data = weight->data();
    std::byte *out_data = out->data();

    for (size_t row = 0; row < index->shape()[0]; ++row) {
        const int64_t embedding_index = index_data[row * index_strides[0]];
        if (embedding_index < 0 ||
            static_cast<size_t>(embedding_index) >= weight->shape()[0]) {
            throw std::out_of_range("embedding index is out of range");
        }

        for (size_t column = 0; column < weight->shape()[1]; ++column) {
            const ptrdiff_t weight_offset =
                embedding_index * weight_strides[0] +
                static_cast<ptrdiff_t>(column) * weight_strides[1];
            const ptrdiff_t out_offset =
                static_cast<ptrdiff_t>(row) * out_strides[0] +
                static_cast<ptrdiff_t>(column) * out_strides[1];
            std::memcpy(out_data + out_offset * static_cast<ptrdiff_t>(element_size),
                        weight_data + weight_offset * static_cast<ptrdiff_t>(element_size),
                        element_size);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void embedding(llaisys::tensor_t out,
               llaisys::tensor_t index,
               llaisys::tensor_t weight) {
    embedding_(out, index, weight);
}

} // namespace llaisys::ops::cpu
