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

import io
import os
from typing import Dict

import numpy as np
import torch

from veturboio.ops.cipher import CipherInfo, decrypt
from veturboio.ops.load_utils import IOHelper, load_file_to_tensor
from veturboio.safetensors import SafetensorsFile
from veturboio.types import FILE_PATH

from .base_loader import PosixLoader


class FasterPosixLoader(PosixLoader):
    def __init__(
        self,
        file: FILE_PATH,
        helper: IOHelper,
        num_thread: int = 32,
        use_pinmem: bool = False,
        use_direct_io: bool = False,
    ) -> None:
        super().__init__(file)
        self.helper = helper
        self.num_thread = num_thread
        self.use_pinmem = use_pinmem
        self.use_direct_io = use_direct_io

    def load_safetensors(
        self, safetensors_file: SafetensorsFile, map_location: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        file_size = os.path.getsize(safetensors_file.file)
        base_offset = safetensors_file.tensor_offset
        device = torch.device(map_location)
        if device.type == "cuda":
            device_id = device.index if device.index is not None else torch.cuda.current_device()
        else:
            device_id = -1

        total_tensor = self.init_aligned_tensor(device, device_id, file_size, base_offset)
        load_file_to_tensor(
            file_path=safetensors_file.file,
            total_tensor=total_tensor,
            sample_tensor=torch.ones([], dtype=torch.uint8),
            offset=base_offset,
            helper=self.helper,
            device_id=device_id,
            num_thread=self.num_thread,
            use_pinmem=self.use_pinmem,
            use_sfcs_sdk=False,
            use_direct_io=self.use_direct_io,
            cipher_info=safetensors_file._cipher_info,
        )

        return SafetensorsFile.split_tensor_to_state_dict(total_tensor, safetensors_file)

    def load_pt(
        self, map_location: str = "cpu", cipher_info: CipherInfo = CipherInfo(False)
    ) -> Dict[str, torch.Tensor]:
        if cipher_info.use_cipher:
            h_off = CipherInfo.HEADER_SIZE if cipher_info.use_header else 0
            arr = np.fromfile(self.file, dtype=np.uint8, offset=h_off, count=-1)
            decrypt(cipher_info, arr, arr, 0)
            return torch.load(io.BytesIO(arr.data), map_location=map_location)

        return torch.load(self.file, map_location=map_location)
