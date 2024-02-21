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

from veturboio.ops.cipher import CipherInfo
from veturboio.ops.sfcs_utils import init_sfcs_conf, sfcs_get_file_size, sfcs_write_file
from veturboio.saver.base_saver import BaseSaver
from veturboio.types import FILE_PATH


class SfcsClientSaver(BaseSaver):
    def __init__(self, use_cipher: bool = False) -> None:
        super().__init__(method="client")

        init_sfcs_conf()

        use_cipher = use_cipher or os.environ.get("VETURBOIO_USE_CIPHER", "0") == "1"
        self.cipher_info = CipherInfo(use_cipher)

    def save_file(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH, metadata: Dict[str, str] = None) -> None:
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
            file_path = tmpfile.name
            safetenors_save_file(state_dict, file_path, metadata=metadata)

            file_size = os.path.getsize(file_path)
            file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)

            sfcs_write_file(file, file_bytes, file_size, self.cipher_info)

    def save_model(self, model: torch.nn.Module, file: FILE_PATH) -> None:
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
            file_path = tmpfile.name
            safetensors_save_model(model, file_path)

            file_size = os.path.getsize(file_path)
            file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)

            sfcs_write_file(file, file_bytes, file_size, self.cipher_info)

    def save_pt(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH) -> None:
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
            file_path = tmpfile.name
            torch.save(state_dict, file_path)

            file_size = os.path.getsize(file_path)
            file_bytes = np.memmap(file_path, dtype=np.byte, mode='r+', shape=file_size)

            sfcs_write_file(file, file_bytes, file_size, self.cipher_info)
