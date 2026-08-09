#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <unordered_set>
namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    
     if (_offset != 0) {
        return false;
    }

    const auto& shape_dims = _meta.shape;
    const auto& stride_dims = _meta.strides;
    int nd = static_cast<int>(ndim());

    // 空张量/0维标量一定连续
    if (nd == 0 || numel() <= 1) {
        return true;
    }

    // 标准连续张量最后一维步长恒等于1
    ptrdiff_t expected_stride = 1;
    // 从最后一维倒序遍历校验stride
    for (int dim = nd - 1; dim >= 0; --dim) {
        if (stride_dims[dim] != expected_stride) {
            return false;
        }
        // 更新下一维期望步长 = 当前维度尺寸 × 当前期望步长
        expected_stride *= static_cast<ptrdiff_t>(shape_dims[dim]);
    }

    // 全部维度步长匹配标准连续布局
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
     const size_t nd = this->ndim();
    const auto &old_shape = _meta.shape;
    const auto &old_strides = _meta.strides;

    // 校验1：order维度数量与张量维度一致
    if (order.size() != nd) {
        throw std::runtime_error(
            "permute error: order length(" + std::to_string(order.size()) +
            ") != tensor ndim(" + std::to_string(nd) + ")"
        );
    }

    // 校验2：维度下标不越界 + 无重复维度
    std::unordered_set<size_t> dim_check;
    for (size_t dim : order) {
        if (dim >= nd) {
            throw std::runtime_error(
                "permute error: dim index " + std::to_string(dim) + " out of range"
            );
        }
        if (dim_check.count(dim)) {
            throw std::runtime_error(
                "permute error: duplicate dimension index " + std::to_string(dim)
            );
        }
        dim_check.insert(dim);
    }

    // 校验3：必须包含全部维度（防止少传维度）
    if (dim_check.size() != nd) {
        throw std::runtime_error("permute error: order missing some dimension indices");
    }

    // 按置换顺序生成新 shape、新 strides
    std::vector<size_t> new_shape;
    std::vector<ptrdiff_t> new_strides;
    new_shape.reserve(nd);
    new_strides.reserve(nd);

    for (size_t src_dim : order) {
        new_shape.push_back(old_shape[src_dim]);
        new_strides.push_back(old_strides[src_dim]);
    }

    // 构造新元数据：dtype不变，替换shape/strides
    TensorMeta new_meta{
        _meta.dtype,
        std::move(new_shape),
        std::move(new_strides)
    };

    // 关键：传入原 offset，不能丢失切片偏移
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
 // 前置：先校验连续性，提前失败，减少无用计算
    if (!this->isContiguous()) {
        throw std::runtime_error(
            "Tensor::view() error: view requires contiguous tensor, call .contiguous() first"
        );
    }

    // 计算目标shape总元素 + 溢出检测
    size_t target_numel = 1;
    const size_t origin_numel = this->numel();
    for (size_t dim : shape) {
        // 溢出判断：相乘后反向校验
        if (dim != 0 && target_numel > origin_numel / dim) {
            throw std::runtime_error("View shape element count overflow, mismatch tensor numel");
        }
        target_numel *= dim;
    }

    if (target_numel != origin_numel) {
        throw std::runtime_error("View shape does not match the number of elements in the tensor.");
    }

    // 计算C连续stride，你的循环逻辑本身是正确的，无需改动
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = 1; i <= shape.size(); i++) {
        size_t dim_idx = shape.size() - i;
        new_strides[dim_idx] = stride;
        stride *= static_cast<ptrdiff_t>(shape[dim_idx]);
    }

    // 复制元信息，替换shape/stride，dtype保持不变
    TensorMeta new_meta{_meta.dtype, shape, new_strides};

    // 关键修复：传入原张量offset，不能丢！
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, this->_offset));

}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    const size_t nd = ndim();
    const auto& old_shape = _meta.shape;
    const auto& old_strides = _meta.strides;
    // 单个元素字节大小
    const size_t elem_bytes = this->elementSize();

    // 校验1：维度不越界
    if (dim >= nd) {
        throw std::runtime_error(
            "slice error: dim " + std::to_string(dim) +
            " out of tensor ndim(" + std::to_string(nd) + ")"
        );
    }

    const size_t dim_len = old_shape[dim];
    // 校验2：切片区间合法性
    if (start >= end) {
        throw std::runtime_error(
            "slice error: start(" + std::to_string(start) + ") >= end(" + std::to_string(end) + ")"
        );
    }
    if (end > dim_len) {
        throw std::runtime_error(
            "slice error: end(" + std::to_string(end) + ") exceeds dim length(" + std::to_string(dim_len) + ")"
        );
    }

    // 1. 构造新 shape：仅修改切片维度长度
    std::vector<size_t> new_shape = old_shape;
    new_shape[dim] = end - start;

    // ========== 修复核心：统一单位为字节 ==========
    // dim_stride：维度上间隔多少个元素
    ptrdiff_t elem_step = old_strides[dim];
    // 跳过 start 个元素，总共偏移多少字节
    size_t byte_delta = static_cast<size_t>(start * elem_step) * elem_bytes;
    size_t new_global_offset = _offset + byte_delta;

    // 2. 构造新 meta：shape 更新，dtype、strides 完全复用原张量
    TensorMeta new_meta{
        _meta.dtype,
        std::move(new_shape),
        old_strides
    };

    // 传入更新后的字节偏移，共享底层 storage
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_global_offset));
}

void Tensor::load(const void *src_) {
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_H2D);

}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
