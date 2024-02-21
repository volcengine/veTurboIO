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

import torch
from loguru import logger

from veturboio.ops.cipher import CipherInfo

try:
    import veturboio_ext

    IOHelper = veturboio_ext.IOHelper
except ImportError:
    IOHelper = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")


def load_file_to_tensor(
    file_path: str,
    total_tensor: torch.Tensor,
    sample_tensor: torch.Tensor,
    offset: int,
    helper: IOHelper,
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
        sample_tensor,
        offset,
        device_id,
        num_thread,
        use_pinmem,
        use_sfcs_sdk,
        use_direct_io,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
    )


def init_io_helper() -> IOHelper:
    return IOHelper()
