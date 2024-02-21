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
from io import BytesIO
from typing import Dict

import numpy as np
import torch
from numpy import ndarray

from veturboio.loader.base_loader import BaseLoader
from veturboio.ops.cipher import CipherInfo
from veturboio.ops.load_utils import IOHelper, load_file_to_tensor
from veturboio.ops.sfcs_utils import init_sfcs_conf, sfcs_get_file_size, sfcs_read_file
from veturboio.safetensors import SafetensorsFile
from veturboio.types import FILE_PATH


class SfcsClientLoader(BaseLoader):
    def __init__(
        self,
        helper: IOHelper,
        num_thread: int = 32,
        use_pinmem: bool = False,
        use_direct_io: bool = False,
        use_cipher: bool = False,
    ) -> None:
        super().__init__(method="client")

        self.helper = helper
        self.num_thread = num_thread
        self.use_pinmem = use_pinmem
        self.use_direct_io = use_direct_io

        use_cipher = use_cipher or os.environ.get("VETURBOIO_USE_CIPHER", "0") == "1"
        self.cipher_info = CipherInfo(use_cipher)

        init_sfcs_conf()

    def load_to_bytes_array(self, file: FILE_PATH, offset: int, count: int) -> ndarray:
        candidate = np.empty([count], dtype=np.byte)
        sfcs_read_file(
            file, candidate, length=count, offset=offset, num_thread=self.num_thread, cipher_info=self.cipher_info
        )
        return candidate

    def load_safetensors(
        self, safetensors_file: SafetensorsFile, map_location: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        file_size = sfcs_get_file_size(safetensors_file.file)
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
            use_sfcs_sdk=True,
            use_direct_io=self.use_direct_io,
            cipher_info=self.cipher_info,
        )

        return SafetensorsFile.split_tensor_to_state_dict(total_tensor, safetensors_file)

    def load_pt(self, file: FILE_PATH, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
        file_size = sfcs_get_file_size(file)
        file_bytes = self.load_to_bytes_array(file, offset=0, count=file_size).tobytes()
        return torch.load(BytesIO(file_bytes), map_location=map_location)
