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
import tempfile
from typing import Any, Dict

import numpy as np
import torch
from safetensors.torch import save_file as safetenors_save_file
from safetensors.torch import save_model as safetensors_save_model

from veturboio.ops.cipher import CipherInfo, CipherMode, create_cipher_with_header
from veturboio.ops.io_utils import IOHelper
from veturboio.ops.io_utils import save_file as fast_save_file
from veturboio.ops.sfcs_utils import (
    init_sfcs_conf,
    path_mapper,
    sfcs_delete_file,
    sfcs_write_file,
    sfcs_write_file_in_parallel,
)
from veturboio.saver.base_saver import BaseSaver
from veturboio.types import FILE_PATH


class SfcsClientSaver(BaseSaver):
    def __init__(
        self,
        file: FILE_PATH,
        helper: IOHelper = None,
        use_cipher: bool = False,
    ) -> None:
        super().__init__(method="client")

        self.file = file
        self.helper = helper

        mount_path = init_sfcs_conf(file)
        self.sfcs_valid_path = path_mapper(self.file, mount_path)

        use_cipher = use_cipher or os.getenv("VETURBOIO_USE_CIPHER", "0") == "1"
        use_header = use_cipher and os.getenv("VETURBOIO_CIPHER_HEADER", "0") == "1"
        if use_header:
            self.cipher_info = create_cipher_with_header(CipherMode.CTR_128, os.path.abspath(self.file))
        else:
            self.cipher_info = CipherInfo(use_cipher, None, os.path.abspath(self.file))

    def save_file(
        self, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str] = None, enable_fast_mode: bool = False
    ) -> None:
        if enable_fast_mode:
            fast_save_file(
                state_dict,
                self.sfcs_valid_path,
                helper=self.helper,
                metadata=metadata,
                cipher_info=self.cipher_info,
                use_sfcs_sdk=True,
            )
        else:
            with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
                file_path = tmpfile.name
                safetenors_save_file(state_dict, file_path, metadata=metadata)

                file_size = os.path.getsize(file_path)
                if self.cipher_info.use_header:
                    h_off = CipherInfo.HEADER_SIZE
                    file_bytes = np.empty(file_size + h_off, dtype=np.byte)
                    file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.byte)
                    file_bytes[h_off:] = np.fromfile(file_path, dtype=np.byte, count=file_size)
                else:
                    file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)
                sfcs_write_file(self.sfcs_valid_path, file_bytes, len(file_bytes), self.cipher_info)

    def save_model(self, model: torch.nn.Module) -> None:
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
            file_path = tmpfile.name
            safetensors_save_model(model, file_path)

            file_size = os.path.getsize(file_path)
            if self.cipher_info.use_header:
                h_off = CipherInfo.HEADER_SIZE
                file_bytes = np.empty(file_size + h_off, dtype=np.byte)
                file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.byte)
                file_bytes[h_off:] = np.fromfile(file_path, dtype=np.byte, count=file_size)
            else:
                file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)
            sfcs_write_file(self.sfcs_valid_path, file_bytes, len(file_bytes), self.cipher_info)

    def save_pt(self, state_dict: Dict[str, torch.Tensor]) -> None:
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
            file_path = tmpfile.name
            torch.save(state_dict, file_path)

            file_size = os.path.getsize(file_path)
            if self.cipher_info.use_header:
                h_off = CipherInfo.HEADER_SIZE
                file_bytes = np.empty(file_size + h_off, dtype=np.byte)
                file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.byte)
                file_bytes[h_off:] = np.fromfile(file_path, dtype=np.byte, count=file_size)
            else:
                file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)
            sfcs_write_file(self.sfcs_valid_path, file_bytes, len(file_bytes), self.cipher_info)
