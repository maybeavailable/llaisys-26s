#pragma once
#include "llaisys.h"

#include <cstddef>

#include "../../../tensor/tensor.hpp"

#include "../../../utils/types.hpp"

namespace llaisys::ops::cpu {
void argmax(llaisys::tensor_t max_idx, llaisys::tensor_t max_val, llaisys::tensor_t vals);

}