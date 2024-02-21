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

import torch
from safetensors.torch import save_file as safetenors_save_file
from safetensors.torch import save_model as safetensors_save_model

from veturboio.types import FILE_PATH


class BaseSaver:
    def __init__(self, method: str) -> None:
        self.method = method

    def save_file(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH, metadata: Dict[str, str] = None) -> None:
        raise NotImplementedError

    def save_model(self, model: torch.nn.Module, file: FILE_PATH) -> None:
        raise NotImplementedError


class PosixSaver(BaseSaver):
    def __init__(self) -> None:
        super().__init__(method="posix")

    def save_file(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH, metadata: Dict[str, str] = None) -> None:
        safetenors_save_file(state_dict, file, metadata=metadata)

    def save_model(self, model: torch.nn.Module, file: FILE_PATH) -> None:
        return safetensors_save_model(model, file)

    def save_pt(self, state_dict: Dict[str, torch.Tensor], file: FILE_PATH) -> None:
        return torch.save(state_dict, file)
