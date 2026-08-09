#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
void self_attention_kernel(llaisys::tensor_t attn_val,
                           llaisys::tensor_t q,
                           llaisys::tensor_t k,
                           llaisys::tensor_t v,
                           float scale) {
    const size_t query_len = q->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t kv_len = k->shape()[0];
    const size_t head_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];
    const size_t head_ratio = query_heads / k->shape()[1];
    const size_t element_size = q->elementSize();
    const auto &q_strides = q->strides();
    const auto &k_strides = k->strides();
    const auto &v_strides = v->strides();
    const auto &out_strides = attn_val->strides();
    const auto *q_data = q->data();
    const auto *k_data = k->data();
    const auto *v_data = v->data();
    auto *out_data = attn_val->data();

    std::vector<float> scores(kv_len);
    for (size_t qi = 0; qi < query_len; ++qi) {
        const size_t last_key = qi + kv_len - query_len;
        for (size_t qh = 0; qh < query_heads; ++qh) {
            const size_t kvh = qh / head_ratio;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t ki = 0; ki < kv_len; ++ki) {
                if (ki > last_key) {
                    scores[ki] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    const auto *q_value = reinterpret_cast<const T *>(
                        q_data + (static_cast<ptrdiff_t>(qi) * q_strides[0] +
                                  static_cast<ptrdiff_t>(qh) * q_strides[1] +
                                  static_cast<ptrdiff_t>(d) * q_strides[2]) *
                                     static_cast<ptrdiff_t>(element_size));
                    const auto *k_value = reinterpret_cast<const T *>(
                        k_data + (static_cast<ptrdiff_t>(ki) * k_strides[0] +
                                  static_cast<ptrdiff_t>(kvh) * k_strides[1] +
                                  static_cast<ptrdiff_t>(d) * k_strides[2]) *
                                     static_cast<ptrdiff_t>(element_size));
                    score += llaisys::utils::cast<float>(*q_value) *
                             llaisys::utils::cast<float>(*k_value);
                }
                scores[ki] = score * scale;
                max_score = std::max(max_score, scores[ki]);
            }

            float denominator = 0.0f;
            for (size_t ki = 0; ki <= last_key; ++ki) {
                scores[ki] = std::exp(scores[ki] - max_score);
                denominator += scores[ki];
            }
            for (size_t d = 0; d < value_dim; ++d) {
                float result = 0.0f;
                for (size_t ki = 0; ki <= last_key; ++ki) {
                    const auto *v_value = reinterpret_cast<const T *>(
                        v_data + (static_cast<ptrdiff_t>(ki) * v_strides[0] +
                                  static_cast<ptrdiff_t>(kvh) * v_strides[1] +
                                  static_cast<ptrdiff_t>(d) * v_strides[2]) *
                                     static_cast<ptrdiff_t>(element_size));
                    result += (scores[ki] / denominator) *
                              llaisys::utils::cast<float>(*v_value);
                }
                auto *output = reinterpret_cast<T *>(
                    out_data + (static_cast<ptrdiff_t>(qi) * out_strides[0] +
                                static_cast<ptrdiff_t>(qh) * out_strides[1] +
                                static_cast<ptrdiff_t>(d) * out_strides[2]) *
                                   static_cast<ptrdiff_t>(element_size));
                *output = llaisys::utils::cast<T>(result);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    if (q->ndim() != 3 || k->ndim() != 3 || v->ndim() != 3 || attn_val->ndim() != 3) {
        throw std::invalid_argument("self_attention expects 3D q, k, v, and output tensors");
    }
    if (q->shape()[2] != k->shape()[2] || k->shape()[0] < q->shape()[0] ||
        q->shape()[1] == 0 || k->shape()[1] == 0 ||
        q->shape()[1] % k->shape()[1] != 0 ||
        attn_val->shape()[0] != q->shape()[0] ||
        attn_val->shape()[1] != q->shape()[1] ||
        attn_val->shape()[2] != v->shape()[2]) {
        throw std::invalid_argument("self_attention tensor shapes are incompatible");
    }
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype() || q->dtype() != attn_val->dtype()) {
        throw std::invalid_argument("self_attention tensors must have the same dtype");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("self_attention scale must be finite");
    }
    if (q->deviceType() != k->deviceType() || q->deviceId() != k->deviceId() ||
        q->deviceType() != v->deviceType() || q->deviceId() != v->deviceId() ||
        q->deviceType() != attn_val->deviceType() || q->deviceId() != attn_val->deviceId()) {
        throw std::invalid_argument("self_attention tensors must be on the same device");
    }

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_kernel<float>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_kernel<llaisys::fp16_t>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_kernel<llaisys::bf16_t>(attn_val, q, k, v, scale);
    default:
        throw std::invalid_argument("self_attention supports only f32, f16, and bf16");
    }
}
}
