from typing import Sequence
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int64, c_size_t, c_void_p, Structure
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType

from pathlib import Path
import json
import safetensors
import torch


class _Qwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t), ("hs", c_size_t), ("nh", c_size_t),
        ("nkvh", c_size_t), ("dh", c_size_t), ("di", c_size_t),
        ("maxseq", c_size_t), ("voc", c_size_t),
        ("epsilon", c_float), ("theta", c_float), ("end_token", c_int64),
    ]


LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
    POINTER(_Qwen2Meta), c_int, POINTER(c_int), c_int
]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p
LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
LIB_LLAISYS.llaisysQwen2ModelLoadWeight.argtypes = [
    c_void_p, c_char_p, c_void_p, POINTER(c_size_t), c_size_t, c_int
]
LIB_LLAISYS.llaisysQwen2ModelLoadWeight.restype = None
LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        dtype_name = config.get("torch_dtype", "bfloat16")
        dtype = DataType.BF16 if dtype_name in ("bfloat16", "bf16") else DataType.F32
        meta = _Qwen2Meta(
            int(dtype), config["num_hidden_layers"], config["hidden_size"],
            config["num_attention_heads"], config["num_key_value_heads"],
            config["hidden_size"] // config["num_attention_heads"],
            config["intermediate_size"], config["max_position_embeddings"],
            config["vocab_size"], config["rms_norm_eps"], config["rope_theta"],
            config.get("eos_token_id", 151643),
        )
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("failed to create Qwen2 model")
        self._meta = meta

        for file in sorted(model_path.glob("*.safetensors")):
            # The reference checkpoint uses bfloat16, which safetensors exposes
            # through torch while preserving the raw two-byte representation.
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                value = data_.get_tensor(name_).contiguous()
                shape = (c_size_t * value.dim())(*value.shape)
                LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                    self._model, name_.encode("utf-8"),
                    c_void_p(value.data_ptr()), shape, value.dim(), int(dtype)
                )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        tokens = [int(token) for token in inputs]
        steps = 128 if max_new_tokens is None else max(0, int(max_new_tokens))
        for _ in range(steps):
            token_array = (c_int64 * len(tokens))(*tokens)
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_array, len(tokens)
            ))
            if next_token < 0:
                raise RuntimeError("Qwen2 inference failed")
            tokens.append(next_token)
            if next_token == self._meta.end_token:
                break
        return tokens

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None
