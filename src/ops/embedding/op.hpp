#pragma once

#include "../../tensor/tensor.hpp"
#include "../../utils/types.hpp"
#include"cpu/embedding.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
