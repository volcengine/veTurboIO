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
from multiprocessing import shared_memory
from typing import Dict, Optional

import torch
from loguru import logger
from safetensors.torch import _remove_duplicate_names

from veturboio.loader import FasterPosixLoader, PosixLoader, SfcsClientLoader
from veturboio.ops.io_utils import IOHelper
from veturboio.safetensors import SafetensorsFile
from veturboio.saver import PosixSaver, SfcsClientSaver
from veturboio.types import FILE_PATH


def is_sfcs_path(file: FILE_PATH):
    if len(file) > 7 and file[:7] == "sfcs://":
        return True, file[6:]
    elif len(file) > 9 and file[:9] == "/dev/shm/":
        return False, file
    elif os.environ.get("VETURBOIO_USE_SFCS_SDK", "0") == "1":
        return True, file
    else:
        return False, file


def load(
    file: FILE_PATH,
    map_location: Optional[str] = "cpu",
    enable_fast_mode: Optional[bool] = True,
    num_thread: Optional[int] = 32,
    helper: Optional[IOHelper] = None,
    use_pinmem: Optional[bool] = False,
    use_direct_io: Optional[bool] = False,
    use_cipher: Optional[bool] = None,
    state_dict: Dict[str, torch.Tensor] = None,
) -> Dict:
    """Load state dict object from checkpoint file. The file can be both safetensors file and pytorch file.
    If the file is safetensors file, it will be loaded by veturboio and the loading speed will be accelerated.

    Args:
        file (FILE_PATH): file path
        map_location (str, optional): map location. Defaults to "cpu".
        enable_fast_mode (bool, optional): enable fast mode. Defaults to True.
        helper (IOHelper, optional): use IOHelper. Defaults to None.
        use_pinmem (bool, optional): use pin memory. Defaults to False.
        num_thread (int, optional): number of threads. Defaults to 32.
        use_direct_io (bool, optional): open file in direct io mode. Defaults to False.
        use_cipher (bool, optional): decrypt file. Defaults to None. Note: cipher is
            disabled by force when use_cipher set to False. Otherwise, when use_cipher
            set to True or environ VETURBOIO_USE_CIPHER set to '1', cipher is enabled.
        state_dict (Dict): pre allocated state dict. Defaults to None.

    Returns:
        state_dict (Dict): state dict

    Examples:
        ```
        import veturboio
        state_dict = veturboio.load("model.safetensors")
        ```
    """

    if IOHelper is None:
        enable_fast_mode = False
    elif helper is None:
        helper = IOHelper()

    use_sfcs_sdk, file = is_sfcs_path(file)
    if enable_fast_mode == False:
        loader = PosixLoader(file)
    elif use_sfcs_sdk:
        loader = SfcsClientLoader(
            helper=helper,
            file=file,
            num_thread=num_thread,
            use_pinmem=use_pinmem,
            use_direct_io=use_direct_io,
        )
    else:
        loader = FasterPosixLoader(
            file,
            helper,
            num_thread=num_thread,
            use_pinmem=use_pinmem,
            use_direct_io=use_direct_io,
        )

    safetensors_file = SafetensorsFile(file, loader, use_cipher)
    return safetensors_file.load(map_location=map_location, state_dict=state_dict)


def load_to_shmem(
    file: FILE_PATH,
    num_thread: Optional[int] = 32,
    helper: Optional[IOHelper] = None,
    use_direct_io: Optional[bool] = False,
    use_cipher: Optional[bool] = None,
) -> shared_memory.SharedMemory:
    """Load checkpoint file to shmem.

    Args:
        file (FILE_PATH): file path
        num_thread (int, optional): number of threads. Defaults to 32.
        helper (IOHelper, optional): use IOHelper. Defaults to None.
        use_cipher (bool, optional): decrypt file. Defaults to None. Note: cipher is
            disabled by force when use_cipher set to False. Otherwise, when use_cipher
            set to True or environ VETURBOIO_USE_CIPHER set to '1', cipher is enabled.

    Returns:
        shmem (shared_memory.SharedMemory): shared memory object.

    Examples:
        ```
        import veturboio
        shmem_file = veturboio.load_to_shmem("sfcs://model.safetensors")
        ```
    """

    if helper is None:
        helper = IOHelper()

    use_sfcs_sdk, file = is_sfcs_path(file)
    if use_sfcs_sdk:
        loader = SfcsClientLoader(
            helper=helper,
            file=file,
            num_thread=num_thread,
        )
    else:
        loader = FasterPosixLoader(
            file,
            helper,
            num_thread=num_thread,
            use_direct_io=use_direct_io,
        )

    safetensors_file = SafetensorsFile(file, loader, use_cipher)
    return safetensors_file.load_to_shmem()


def save_file(
    state_dict: Dict[str, torch.Tensor],
    file: FILE_PATH,
    force_contiguous: bool = True,
    force_save_shared_tensor: bool = False,
    force_clone_shared_tensor: bool = False,
    metadata: Dict[str, str] = None,
    use_cipher: Optional[bool] = False,
    helper: Optional[IOHelper] = None,
    enable_fast_mode: Optional[bool] = False,
) -> None:
    """Save state dict object to safetensors file.

    Args:
        state_dict (Dict): state dict
        file (FILE_PATH): file path
        force_contiguous (bool, optional): force contiguous. Defaults to True.
        force_save_shared_tensor (bool, optional): force save shared tensor. Defaults to False.
        force_clone_shared_tensor (bool, optional): force to clone shared tensor rather than delete
            when force_save_shared_tensor is enabled. Defaults to False.
        metadata (Dict[str, str], optional): metadata. Defaults to None.
        use_cipher (bool, optional): decrypt file. Defaults to False.
        helper (IOHelper, optional): use IOHelper. Defaults to None.
        enable_fast_mode (bool, optional): enable fast mode. Defaults to False.

    Examples:
        ```
        import torch
        import veturboio

        state_dict = {"weight": torch.randn(10, 10)}
        veturboio.save_file(state_dict, "model.safetensors")
        ```
    """
    if helper is None:
        helper = IOHelper()

    use_sfcs_sdk, file = is_sfcs_path(file)
    if use_sfcs_sdk:
        saver = SfcsClientSaver(file=file, use_cipher=use_cipher, helper=helper)
    else:
        saver = PosixSaver(file=file, use_cipher=use_cipher, helper=helper)

    # TODO: there are some bugs while state_dict is loaded from veturboio
    if not force_save_shared_tensor:
        if force_clone_shared_tensor:
            logger.warning("force_clone_shared_tensor won't take any effect while force_save_shared_tensor is False;")
        try:
            saver.save_file(state_dict, metadata=metadata, enable_fast_mode=enable_fast_mode)
        except ValueError as e:
            msg = str(e)
            raise ValueError(msg)
        else:
            return

    to_removes = _remove_duplicate_names(state_dict)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if metadata is None:
                metadata = {}

            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            if force_clone_shared_tensor:
                state_dict[to_remove] = state_dict[to_remove].clone()
            else:
                del state_dict[to_remove]
    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    return saver.save_file(state_dict, metadata=metadata, enable_fast_mode=enable_fast_mode)


def save_model(model: torch.nn.Module, file: FILE_PATH, use_cipher: Optional[bool] = False) -> None:
    """Save model state dict to safetensors file.

    Args:
        model (torch.nn.Module): model
        file (FILE_PATH): file path
        use_cipher (bool, optional): decrypt file. Defaults to False.

    Examples:
        ```
        import torch
        import veturboio

        model = torch.nn.Linear(10, 10)
        veturboio.save_model(model, "model.safetensors")
        ```
    """

    use_sfcs_sdk, file = is_sfcs_path(file)
    if use_sfcs_sdk:
        saver = SfcsClientSaver(file=file, use_cipher=use_cipher)
    else:
        saver = PosixSaver(file=file, use_cipher=use_cipher)

    return saver.save_model(model)


def save_pt(state_dict: Dict[str, torch.Tensor], file: FILE_PATH, use_cipher: Optional[bool] = False) -> None:
    """Save state dict object to pytorch file.

    Args:
        state_dict (Dict): state dict
        file (FILE_PATH): file path
        use_cipher (bool, optional): encrypt file. Defaults to False.

    Examples:
        ```
        import torch
        import veturboio

        state_dict = {"weight": torch.randn(10, 10)}
        veturboio.save_pt(state_dict, "model.pt")
        ```
    """
    use_sfcs_sdk, file = is_sfcs_path(file)
    if use_sfcs_sdk:
        saver = SfcsClientSaver(file=file, use_cipher=use_cipher)
    else:
        saver = PosixSaver(file=file, use_cipher=use_cipher)

    return saver.save_pt(state_dict)
