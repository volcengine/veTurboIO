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

from typing import Any, Dict

import numpy as np
import torch
from numpy import ndarray

# from veturboio.safetensors import SafetensorsFile
from veturboio.types import FILE_PATH

SAFETENSORS_FILE_MAGIC_NUM = 123
BUF_ALIGN_SIZE = 4096


class BaseLoader:
    def __init__(self, method: str) -> None:
        self.method = method

    def load_to_bytes_array(self, file: FILE_PATH, offset: int, count: int) -> ndarray:
        raise NotImplementedError

    def load_safetensors(self, safetensors_file: Any, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def init_aligned_tensor(self, device, device_id: int, file_size, base_offset: int) -> torch.Tensor:
        if device_id != -1:
            try:
                total_tensor = torch.empty(file_size - base_offset, dtype=torch.uint8, device=device)
            except RuntimeError as e:
                msg = str(e)
                raise RuntimeError(msg)

        else:
            array = np.empty(file_size - base_offset + BUF_ALIGN_SIZE, dtype=np.uint8)
            offset1 = array.ctypes.data % BUF_ALIGN_SIZE
            offset2 = base_offset % BUF_ALIGN_SIZE
            if offset1 > offset2:
                align = BUF_ALIGN_SIZE - offset1 + offset2
            else:
                align = offset2 - offset1

            sub_array = array[align : align + file_size - base_offset].view(dtype=np.uint8)
            total_tensor = torch.from_numpy(sub_array)
        return total_tensor


class PosixLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__(method="posix")

    def load_to_bytes_array(self, file: FILE_PATH, offset: int, count: int) -> ndarray:
        return np.fromfile(file, dtype=np.byte, offset=offset, count=count)

    def load_safetensors(self, safetensors_file: Any, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
        state_dict = {}

        base_offset = safetensors_file.tensor_offset
        device = torch.device(map_location)

        for tensor_meta in safetensors_file.meta.values():
            tensor_bytes = np.memmap(
                safetensors_file.file,
                dtype=np.byte,
                mode="r",
                offset=base_offset + tensor_meta.data_offsets[0],
                shape=tensor_meta.data_offsets[1] - tensor_meta.data_offsets[0],
            )
            tensor = torch.frombuffer(tensor_bytes, dtype=tensor_meta.dtype)
            tensor = tensor.view(tensor_meta.shape)
            if device.type == "cuda":
                state_dict[tensor_meta.name] = tensor.pin_memory().to(device=device, non_blocking=True)
            else:
                state_dict[tensor_meta.name] = tensor

        return state_dict

    def load_pt(self, file: FILE_PATH, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
        return torch.load(file, map_location=map_location)
