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

from typing import Optional

import numpy as np
from loguru import logger

from veturboio.ops.cipher import CipherInfo

try:
    from veturboio.utils.load_veturboio_ext import load_veturboio_ext

    veturboio_ext = load_veturboio_ext()
    IOHelper = veturboio_ext.IOHelper
    POSIXFile = veturboio_ext.POSIXFile
except ImportError:
    POSIXFile = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")


def posix_read_file(
    file_path: str,
    arr: np.ndarray,
    length: int,
    offset: int,
    num_thread: Optional[int] = 1,
    cipher_info: CipherInfo = CipherInfo(False),
    use_direct_io: bool = False,
) -> int:
    posix_file = POSIXFile(
        file_path,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )
    return posix_file.read_file_to_array(arr, length, offset, num_thread, use_direct_io)
