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

import os
import random
import string
from io import BytesIO
from multiprocessing import shared_memory
from typing import Dict

import numpy as np
import torch
from numpy import ndarray

from veturboio.loader.base_loader import BaseLoader
from veturboio.ops.cipher import CipherInfo
from veturboio.ops.io_utils import IOHelper, load_file_to_tensor
from veturboio.ops.sfcs_utils import (
    init_sfcs_conf,
    path_mapper,
    sfcs_default_config,
    sfcs_get_file_size,
    sfcs_read_file,
)
from veturboio.safetensors import SafetensorsFile
from veturboio.types import FILE_PATH


class SfcsClientLoader(BaseLoader):
    def __init__(
        self,
        file: FILE_PATH,
        helper: IOHelper,
        num_thread: int = 32,
        use_pinmem: bool = False,
        use_direct_io: bool = False,
    ) -> None:
        super().__init__(method="client")

        self.file = file
        self.helper = helper
        self.num_thread = num_thread
        self.use_pinmem = use_pinmem
        self.use_direct_io = use_direct_io
        self._mount_path = init_sfcs_conf(file)
        self._sfcs_valid_path = path_mapper(self.file, self._mount_path)

    def load_to_bytes(self, offset: int, count: int, cipher_info: CipherInfo = CipherInfo(False)) -> bytes:
        file_size = sfcs_get_file_size(self._sfcs_valid_path)
        if offset + count > file_size:
            count = file_size - offset

        file_bytes = bytes(count)
        candidate = np.frombuffer(file_bytes, dtype=np.byte)
        sfcs_read_file(
            self._sfcs_valid_path,
            candidate,
            length=count,
            offset=offset,
            num_thread=self.num_thread,
            cipher_info=cipher_info,
        )
        return file_bytes

    def load_to_shmem(self, cipher_info: CipherInfo = CipherInfo(False)) -> shared_memory.SharedMemory:
        file_size = sfcs_get_file_size(self._sfcs_valid_path)
        file_name = ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase, 10))
        shm = shared_memory.SharedMemory(name=file_name, create=True, size=file_size)

        h_off = CipherInfo.HEADER_SIZE if cipher_info.use_header else 0
        candidate = np.frombuffer(shm.buf, dtype=np.byte)
        sfcs_read_file(
            self._sfcs_valid_path,
            candidate,
            length=file_size - h_off,
            offset=h_off,
            num_thread=self.num_thread,
            cipher_info=cipher_info,
        )
        return shm

    def load_safetensors(
        self,
        safetensors_file: SafetensorsFile,
        map_location: str = "cpu",
        state_dict: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # TODO should be the same as self.loader
        sfcs_valid_path = path_mapper(safetensors_file.file, self._mount_path)
        file_size = sfcs_get_file_size(sfcs_valid_path)
        base_offset = safetensors_file.tensor_offset
        device = torch.device(map_location)
        if device.type == "cuda":
            device_id = device.index if device.index is not None else torch.cuda.current_device()
        else:
            device_id = -1

        if state_dict:
            for tensor_meta in safetensors_file._meta.values():
                tensor = state_dict[tensor_meta.name]
                if not tensor.is_contiguous():
                    raise RuntimeError("allocated tensor not contiguous")
                if not tensor.dtype == tensor_meta.dtype:
                    raise RuntimeError("allocated tensor dtype not match")

                offset = tensor_meta.data_offsets[0]
                length = tensor_meta.data_offsets[1] - tensor_meta.data_offsets[0]
                tensor_length = torch.numel(tensor) * tensor.element_size()
                if tensor_length < length:
                    raise RuntimeError("allocated tensor size not enough")

                load_file_to_tensor(
                    file_path=sfcs_valid_path,
                    total_tensor=tensor,
                    length=length,
                    offset=base_offset + offset,
                    helper=self.helper,
                    device_id=device_id,
                    num_thread=self.num_thread,
                    use_pinmem=self.use_pinmem,
                    use_sfcs_sdk=True,
                    use_direct_io=self.use_direct_io,
                    cipher_info=safetensors_file._cipher_info,
                )
                tensor = tensor.resize_(tensor_meta.shape)
                state_dict[tensor_meta.name] = tensor
            return state_dict
        else:
            total_tensor = self.init_aligned_tensor(device, device_id, file_size, base_offset)
            load_file_to_tensor(
                file_path=sfcs_valid_path,
                total_tensor=total_tensor,
                offset=base_offset,
                helper=self.helper,
                device_id=device_id,
                num_thread=self.num_thread,
                use_pinmem=self.use_pinmem,
                use_sfcs_sdk=True,
                use_direct_io=self.use_direct_io,
                cipher_info=safetensors_file._cipher_info,
            )

        return SafetensorsFile.split_tensor_to_state_dict(total_tensor, safetensors_file)

    def load_pt(
        self, map_location: str = "cpu", cipher_info: CipherInfo = CipherInfo(False)
    ) -> Dict[str, torch.Tensor]:
        file_size = sfcs_get_file_size(self._sfcs_valid_path)
        h_off = CipherInfo.HEADER_SIZE if cipher_info.use_header else 0
        file_bytes = self.load_to_bytes(offset=h_off, count=file_size - h_off, cipher_info=cipher_info)
        return torch.load(BytesIO(file_bytes), map_location=map_location)
