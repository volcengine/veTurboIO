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

from veturboio.ops.cipher import CipherInfo, CipherMode, create_cipher_with_header, encrypt
from veturboio.types import FILE_PATH


class BaseSaver:
    def __init__(self, method: str) -> None:
        self.method = method

    def save_file(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH, metadata: Dict[str, str] = None) -> None:
        raise NotImplementedError

    def save_model(self, model: torch.nn.Module, file: FILE_PATH) -> None:
        raise NotImplementedError


class PosixSaver(BaseSaver):
    def __init__(self, file: FILE_PATH, use_cipher: bool = False) -> None:
        super().__init__(method="posix")
        self.file = file
        use_cipher = use_cipher or os.getenv("VETURBOIO_USE_CIPHER", "0") == "1"
        use_header = use_cipher and os.getenv("VETURBOIO_CIPHER_HEADER", "0") == "1"
        if use_header:
            self.cipher_info = create_cipher_with_header(CipherMode.CTR_128)
        else:
            self.cipher_info = CipherInfo(use_cipher)

    def save_file(self, state_dict: Dict[str, torch.Tensor], metadata: Dict[str, str] = None) -> None:
        if self.cipher_info.use_cipher:
            with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
                tmp_file_path = tmpfile.name
                safetenors_save_file(state_dict, tmp_file_path, metadata=metadata)
                tmp_file_size = os.path.getsize(tmp_file_path)
                tmp_file_bytes = np.memmap(tmp_file_path, dtype=np.uint8, mode='r', shape=tmp_file_size)
                h_off = CipherInfo.HEADER_SIZE if self.cipher_info.use_header else 0
                file_bytes = np.memmap(self.file, dtype=np.uint8, mode='w+', shape=tmp_file_size + h_off)
                encrypt(self.cipher_info, tmp_file_bytes, file_bytes[h_off:], 0)
                if h_off:
                    file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.uint8)
                file_bytes.flush()
        else:
            safetenors_save_file(state_dict, self.file, metadata=metadata)

    def save_model(self, model: torch.nn.Module) -> None:
        if self.cipher_info.use_cipher:
            with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
                tmp_file_path = tmpfile.name
                safetensors_save_model(model, tmp_file_path)
                tmp_file_size = os.path.getsize(tmp_file_path)
                tmp_file_bytes = np.memmap(tmp_file_path, dtype=np.uint8, mode='r', shape=tmp_file_size)
                h_off = CipherInfo.HEADER_SIZE if self.cipher_info.use_header else 0
                file_bytes = np.memmap(self.file, dtype=np.uint8, mode='w+', shape=tmp_file_size + h_off)
                encrypt(self.cipher_info, tmp_file_bytes, file_bytes[h_off:], 0)
                if h_off:
                    file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.uint8)
                file_bytes.flush()
        else:
            safetensors_save_model(model, self.file)

    def save_pt(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if self.cipher_info.use_cipher:
            with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpfile:
                tmp_file_path = tmpfile.name
                torch.save(state_dict, tmp_file_path)
                tmp_file_size = os.path.getsize(tmp_file_path)
                tmp_file_bytes = np.memmap(tmp_file_path, dtype=np.uint8, mode='r', shape=tmp_file_size)
                h_off = CipherInfo.HEADER_SIZE if self.cipher_info.use_header else 0
                file_bytes = np.memmap(self.file, dtype=np.uint8, mode='w+', shape=tmp_file_size + h_off)
                encrypt(self.cipher_info, tmp_file_bytes, file_bytes[h_off:], 0)
                if h_off:
                    file_bytes[:h_off] = np.frombuffer(self.cipher_info.to_header_bytes(), dtype=np.uint8)
                file_bytes.flush()
        else:
            torch.save(state_dict, self.file)
