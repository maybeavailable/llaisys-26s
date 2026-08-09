#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    std::unordered_map<std::string, llaisys::tensor_t> weights;
    LlaisysQwen2Weights public_weights{};
    size_t cache_len = 0;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;
};

namespace {

llaisys::tensor_t tensor(const std::vector<size_t> &shape,
                         llaisysDataType_t dtype,
                         LlaisysQwen2Model *model) {
    return llaisys::Tensor::create(shape, dtype, model->device, model->device_id);
}

llaisys::tensor_t get_weight(LlaisysQwen2Model *model, const std::string &name) {
    auto it = model->weights.find(name);
    if (it == model->weights.end()) {
        throw std::runtime_error("missing Qwen2 weight: " + name);
    }
    return it->second;
}

void add_inplace(llaisys::tensor_t dst, llaisys::tensor_t value) {
    llaisys::ops::add(dst, dst, value);
}

} // namespace

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                           llaisysDeviceType_t device,
                                           int *device_ids,
                                           int ndevice) {
    try {
        if (meta == nullptr || ndevice < 1 || device_ids == nullptr) {
            throw std::invalid_argument("invalid Qwen2 model creation arguments");
        }
        auto *model = new LlaisysQwen2Model;
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids[0];
        return model;
    } catch (...) {
        return nullptr;
    }
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return model == nullptr ? nullptr : &model->public_weights;
}

void llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model,
                                 const char *name,
                                 const void *data,
                                 size_t *shape,
                                 size_t ndim,
                                 llaisysDataType_t dtype) {
    try {
        if (model == nullptr || name == nullptr || data == nullptr || shape == nullptr || ndim == 0) {
            throw std::invalid_argument("invalid Qwen2 weight arguments");
        }
        std::vector<size_t> shape_vec(shape, shape + ndim);
        auto value = tensor(shape_vec, dtype, model);
        value->load(data);
        model->weights[name] = std::move(value);
    } catch (...) {
    }
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model,
                               int64_t *token_ids,
                               size_t ntoken) {
    try {
        if (model == nullptr || token_ids == nullptr || ntoken == 0) {
            throw std::invalid_argument("invalid Qwen2 inference arguments");
        }
    if (ntoken > model->meta.maxseq) {
        throw std::invalid_argument("Qwen2 sequence exceeds maxseq");
    }
    const bool incremental = model->cache_len != 0 && ntoken == model->cache_len + 1;
    const size_t token_offset = incremental ? model->cache_len : 0;
    const size_t run_len = incremental ? 1 : ntoken;
    if (!incremental) {
        model->cache_len = 0;
        model->k_cache.clear();
        model->v_cache.clear();
    }
    const auto dtype = model->meta.dtype;
    auto ids = tensor({run_len}, LLAISYS_DTYPE_I64, model);
    ids->load(token_ids + token_offset);
    auto hidden = tensor({run_len, model->meta.hs}, dtype, model);
    llaisys::ops::embedding(hidden, ids, get_weight(model, "model.embed_tokens.weight"));

    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        auto norm = tensor({run_len, model->meta.hs}, dtype, model);
        llaisys::ops::rms_norm(norm, hidden, get_weight(model, prefix + "input_layernorm.weight"), model->meta.epsilon);

        auto q_flat = tensor({run_len, model->meta.nh * model->meta.dh}, dtype, model);
        auto k_flat = tensor({run_len, model->meta.nkvh * model->meta.dh}, dtype, model);
        auto v_flat = tensor({run_len, model->meta.nkvh * model->meta.dh}, dtype, model);
        auto q_bias = get_weight(model, prefix + "self_attn.q_proj.bias");
        auto k_bias = get_weight(model, prefix + "self_attn.k_proj.bias");
        auto v_bias = get_weight(model, prefix + "self_attn.v_proj.bias");
        llaisys::ops::linear(q_flat, norm, get_weight(model, prefix + "self_attn.q_proj.weight"), q_bias);
        llaisys::ops::linear(k_flat, norm, get_weight(model, prefix + "self_attn.k_proj.weight"), k_bias);
        llaisys::ops::linear(v_flat, norm, get_weight(model, prefix + "self_attn.v_proj.weight"), v_bias);

        auto q = q_flat->view({run_len, model->meta.nh, model->meta.dh});
        auto k = k_flat->view({run_len, model->meta.nkvh, model->meta.dh});
        auto v = v_flat->view({run_len, model->meta.nkvh, model->meta.dh});
        auto positions = tensor({run_len}, LLAISYS_DTYPE_I64, model);
        std::vector<int64_t> position_ids(run_len);
        for (size_t position = 0; position < run_len; ++position) {
            position_ids[position] = static_cast<int64_t>(token_offset + position);
        }
        positions->load(position_ids.data());
        auto q_rot = tensor({run_len, model->meta.nh, model->meta.dh}, dtype, model);
        auto k_rot = tensor({run_len, model->meta.nkvh, model->meta.dh}, dtype, model);
        llaisys::ops::rope(q_rot, q, positions, model->meta.theta);
        llaisys::ops::rope(k_rot, k, positions, model->meta.theta);
        if (model->k_cache.size() <= layer) {
            model->k_cache.push_back(k_rot);
            model->v_cache.push_back(v);
        } else {
            const size_t total_len = model->cache_len + run_len;
            auto new_k = tensor({total_len, model->meta.nkvh, model->meta.dh}, dtype, model);
            auto new_v = tensor({total_len, model->meta.nkvh, model->meta.dh}, dtype, model);
            const size_t row_bytes = model->meta.nkvh * model->meta.dh * new_k->elementSize();
            std::memcpy(new_k->data(), model->k_cache[layer]->data(), model->cache_len * row_bytes);
            std::memcpy(new_v->data(), model->v_cache[layer]->data(), model->cache_len * row_bytes);
            std::memcpy(new_k->data() + model->cache_len * row_bytes, k_rot->data(), run_len * row_bytes);
            std::memcpy(new_v->data() + model->cache_len * row_bytes, v->data(), run_len * row_bytes);
            model->k_cache[layer] = std::move(new_k);
            model->v_cache[layer] = std::move(new_v);
        }
        auto attn = tensor({run_len, model->meta.nh, model->meta.dh}, dtype, model);
        llaisys::ops::self_attention(attn, q_rot, model->k_cache[layer], model->v_cache[layer], 1.0f / std::sqrt(static_cast<float>(model->meta.dh)));
        auto attn_flat = attn->view({run_len, model->meta.hs});
        auto projected = tensor({run_len, model->meta.hs}, dtype, model);
        llaisys::ops::linear(projected, attn_flat, get_weight(model, prefix + "self_attn.o_proj.weight"), nullptr);
        add_inplace(hidden, projected);

        auto mlp_norm = tensor({run_len, model->meta.hs}, dtype, model);
        llaisys::ops::rms_norm(mlp_norm, hidden, get_weight(model, prefix + "post_attention_layernorm.weight"), model->meta.epsilon);
        auto gate = tensor({run_len, model->meta.di}, dtype, model);
        auto up = tensor({run_len, model->meta.di}, dtype, model);
        llaisys::ops::linear(gate, mlp_norm, get_weight(model, prefix + "mlp.gate_proj.weight"), nullptr);
        llaisys::ops::linear(up, mlp_norm, get_weight(model, prefix + "mlp.up_proj.weight"), nullptr);
        auto mlp = tensor({run_len, model->meta.di}, dtype, model);
        llaisys::ops::swiglu(mlp, gate, up);
        auto down = tensor({run_len, model->meta.hs}, dtype, model);
        llaisys::ops::linear(down, mlp, get_weight(model, prefix + "mlp.down_proj.weight"), nullptr);
        add_inplace(hidden, down);
    }

    model->cache_len = ntoken;
    auto final_norm = tensor({run_len, model->meta.hs}, dtype, model);
    llaisys::ops::rms_norm(final_norm, hidden, get_weight(model, "model.norm.weight"), model->meta.epsilon);
    auto logits = tensor({run_len, model->meta.voc}, dtype, model);
    llaisys::ops::linear(logits, final_norm, get_weight(model, "lm_head.weight"), nullptr);
    const size_t row_bytes = model->meta.voc * logits->elementSize();
    auto last = tensor({model->meta.voc}, dtype, model);
    std::memcpy(last->data(), logits->data() + (run_len - 1) * row_bytes, row_bytes);
    auto index = tensor({1}, LLAISYS_DTYPE_I64, model);
    auto value = tensor({1}, dtype, model);
    llaisys::ops::argmax(index, value, last);
        return *reinterpret_cast<const int64_t *>(index->data());
    } catch (...) {
        return -1;
    }
}

} // extern "C"
