'''
Copyright (c) 2024 Beijing Volcano Engine Technology Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import json
import os
import pprint
from typing import Callable, Dict, List

import numpy as np
import torch
from loguru import logger

from veturboio.loader import BaseLoader
from veturboio.ops.cipher import CipherInfo
from veturboio.types import FILE_PATH

# All safetensors file will start with a json string, which is the meta info of the file.
# We use the beginning char to determine whether it is a safetensors file. The beginning
# char is '{' and its ascii code is 123.
SAFETENSORS_FILE_MAGIC_NUM = 123

_safetensors_dtype_mapper = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def only_safetensors_property(func: Callable):
    func_name = func.__name__
    warning_msg = "This safetensors file is invalid, will take it as a normal torch file."

    def wrapper(self, *args, **kwargs):
        if not self.is_valid:
            logger.patch(lambda r: r.update(function=func_name)).warning(warning_msg)
            return None
        return func(self, *args, **kwargs)

    return wrapper


class TensorMeta:
    def __init__(self, name: str, dtype: str, shape: List[int], data_offsets: List[int]) -> None:
        self._name = name
        self._dtype = _safetensors_dtype_mapper[dtype]
        self._shape = shape
        self._data_offsets = data_offsets

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def data_offsets(self) -> List[int]:
        return self._data_offsets

    def __str__(self) -> str:
        return str(
            {
                "name": self._name,
                "dtype": self._dtype,
                "shape": self._shape,
                "data_offsets": self._data_offsets,
            }
        )

    def __repr__(self) -> str:
        return self.__str__()


class SafetensorsFile:
    def __init__(self, file: FILE_PATH, loader: BaseLoader, use_cipher: bool = False) -> None:
        self._file = file
        self._loader = loader

        self._is_valid = True

        # cipher related
        self._cipher_info = CipherInfo(False)
        if use_cipher or os.getenv("VETURBOIO_USE_CIPHER", "0") == "1":
            header_bytes = loader.load_to_bytes(file, offset=0, count=CipherInfo.HEADER_SIZE)
            self._cipher_info = CipherInfo(True, header_bytes)

        if self._cipher_info.use_header:
            h_off = CipherInfo.HEADER_SIZE
        else:
            h_off = 0

        magic_number = loader.load_to_bytes(file, offset=8 + h_off, count=1, cipher_info=self._cipher_info)[0]
        if magic_number != SAFETENSORS_FILE_MAGIC_NUM:
            self._is_valid = False
            return

        self._meta_size = np.frombuffer(
            loader.load_to_bytes(file, offset=h_off, count=8, cipher_info=self._cipher_info), dtype=np.int64
        )[0]
        meta_bytes = loader.load_to_bytes(file, offset=8 + h_off, count=self._meta_size, cipher_info=self._cipher_info)
        meta_dict = json.loads(meta_bytes.decode("utf-8"))

        self._shared_tensor = {}
        self._ignored_meta = {}
        if "__metadata__" in meta_dict:
            meta_data = meta_dict.pop("__metadata__")
            for key, value in meta_data.items():
                if value not in meta_dict:
                    self._ignored_meta[key] = value
                else:
                    self._shared_tensor[key] = value

        self._meta = {}
        for key in meta_dict:
            self._meta[key] = TensorMeta(
                name=key,
                dtype=meta_dict[key]["dtype"],
                shape=meta_dict[key]["shape"],
                data_offsets=meta_dict[key]["data_offsets"],
            )

        # record the offset of the tensor data
        self._tensor_offset = np.dtype(np.int64).itemsize + self._meta_size + h_off

    @staticmethod
    def split_tensor_to_state_dict(
        total_tensor: torch.Tensor, safetensor_file: "SafetensorsFile"
    ) -> Dict[str, torch.Tensor]:
        state_dict = {}

        for tensor_meta in safetensor_file.meta.values():
            tensor = total_tensor[tensor_meta.data_offsets[0] : tensor_meta.data_offsets[1]]
            tensor = tensor.view(dtype=tensor_meta.dtype)
            tensor = tensor.reshape(tensor_meta.shape)
            state_dict[tensor_meta.name] = tensor

        for src_tensor_key, tgt_tensor_key in safetensor_file.shared_tensor.items():
            state_dict[src_tensor_key] = state_dict[tgt_tensor_key]
        return state_dict

    @property
    def file(self) -> FILE_PATH:
        return self._file

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    @only_safetensors_property
    def meta_size(self) -> int:
        return self._meta_size

    @property
    @only_safetensors_property
    def meta(self) -> Dict[str, TensorMeta]:
        return self._meta

    @property
    @only_safetensors_property
    def tensor_offset(self) -> int:
        return self._tensor_offset

    @property
    @only_safetensors_property
    def shared_tensor(self) -> Dict[str, str]:
        return self._shared_tensor

    def __str__(self) -> str:
        if not self._is_valid:
            return f"{self.file} is not a valid safetensors file."
        return pprint.pformat(
            {
                "file": self._file,
                "meta_size": self._meta_size,
                "meta": self._meta,
                "tensor_offset": self._tensor_offset,
            }
        )

    def __repr__(self) -> str:
        return self.__str__()

    def load(self, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
        if not self._is_valid:
            return self._loader.load_pt(self.file, map_location, self._cipher_info)
        else:
            return self._loader.load_safetensors(self, map_location)
