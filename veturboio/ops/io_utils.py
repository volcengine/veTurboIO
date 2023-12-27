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
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger
from safetensors.torch import save_file as safetensors_save_file

from veturboio.ops.cipher import CipherInfo, CipherMode, create_cipher_with_header, encrypt
from veturboio.ops.sfcs_utils import sfcs_delete_file, sfcs_write_file, sfcs_write_file_in_parallel
from veturboio.safetensors import parse_state_dict
from veturboio.types import FILE_PATH

try:
    import veturboio_ext

    IOHelper = veturboio_ext.IOHelper
except ImportError:
    IOHelper = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")


def load_file_to_tensor(
    file_path: str,
    total_tensor: torch.Tensor,
    offset: int,
    helper: IOHelper,
    length: int = 0,
    device_id: Optional[int] = -1,
    num_thread: Optional[int] = 32,
    use_pinmem: Optional[bool] = False,
    use_sfcs_sdk: Optional[bool] = False,
    use_direct_io: Optional[bool] = False,
    cipher_info: CipherInfo = CipherInfo(False),
) -> torch.Tensor:
    return helper.load_file_to_tensor(
        file_path,
        total_tensor,
        length,
        offset,
        device_id,
        num_thread,
        use_pinmem,
        use_sfcs_sdk,
        use_direct_io,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )


def save_tensor_to_file(
    tensor: torch.Tensor,
    file_path: FILE_PATH,
    length: int,
    helper: IOHelper,
    use_pinmem: Optional[bool] = False,
    use_sfcs_sdk: Optional[bool] = False,
    cipher_info: CipherInfo = CipherInfo(False),
):
    return helper.save_tensor_to_file(
        tensor,
        file_path,
        length,
        use_pinmem,
        use_sfcs_sdk,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )


def save_file(
    state_dict: Dict[str, torch.Tensor],
    filename: FILE_PATH,
    helper: IOHelper,
    metadata: Optional[Dict[str, str]] = None,
    use_sfcs_sdk: bool = False,
    cipher_info: CipherInfo = CipherInfo(False),
):
    if helper is None:
        if cipher_info.use_cipher:
            logger.warning("helper is None, cipher is not supported in pure python implementation")
        return safetensors_save_file(state_dict, filename, metadata=metadata)

    meta, tensors, sizes, offsets = parse_state_dict(state_dict)

    if metadata:
        meta["__metadata__"] = metadata

    meta_bytes = json.dumps(meta).encode('utf-8')
    meta_len = len(meta_bytes)

    # alignment
    if not meta_len % 8 == 0:
        meta_len_pad = (meta_len + 8) // 8 * 8
        meta_bytes += b' ' * (meta_len_pad - meta_len)
        meta_len = meta_len_pad

    st_header_bytes = meta_len.to_bytes(8, 'little') + meta_bytes
    st_header_len = len(st_header_bytes)

    if use_sfcs_sdk:
        sfcs_write_file_in_parallel(filename, tensors, sizes, offsets, st_header_bytes, st_header_len, cipher_info)
    else:
        with open(filename, "wb") as f:
            if cipher_info.use_cipher:
                if cipher_info.use_header:
                    cipher_header_bytes = cipher_info.to_header_bytes()
                    f.write(cipher_header_bytes)
                enc_st_header_arr = np.zeros(st_header_len, dtype=np.uint8)
                encrypt(cipher_info, np.frombuffer(st_header_bytes, dtype=np.uint8), enc_st_header_arr, 0)
                f.write(enc_st_header_arr.tobytes())
            else:
                f.write(st_header_bytes)

        for i in range(len(tensors)):
            tensor = tensors[i]
            size = sizes[i]
            save_tensor_to_file(
                tensor,
                filename,
                size,
                helper=helper,
                use_pinmem=False,
                use_sfcs_sdk=use_sfcs_sdk,
                cipher_info=cipher_info,
            )


def init_io_helper() -> IOHelper:
    return IOHelper()
