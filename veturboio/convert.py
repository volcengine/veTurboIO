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

import argparse
import gc
import logging
import os
import sys
import traceback
from datetime import datetime

import torch
from safetensors.torch import _find_shared_tensors, _is_complete

import veturboio


def to_valid_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    invalid_key = [k for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
    if len(invalid_key) > 0:
        logger.warning(f"invalid keys to removed: {invalid_key}")
        state_dict = {k: v for k, v in state_dict.items() if k not in invalid_key}

    result = {}
    shared_tensor_groups = _find_shared_tensors(state_dict)
    for group in shared_tensor_groups:
        # check if all share tensors have the same data ptr, same shape, and same size
        shared_tensors = [state_dict[k] for k in group]
        data_ptrs = [t.data_ptr() for t in shared_tensors]
        shapes = [t.shape for t in shared_tensors]
        if len(set(data_ptrs)) != 1 or len(set(shapes)) != 1:
            raise Exception(f"shared tensors {group} are not equal")
        # make sure these tensors are complete and identical
        converted_tensor = shared_tensors[0]
        if not _is_complete(converted_tensor):
            converted_tensor = converted_tensor.clone()
        for t in group:
            result[t] = converted_tensor
    for k, v in state_dict.items():
        if k not in result:
            result[k] = v
    return result


def add_handlers(logger: logging.Logger):
    """
    Add handlers to logger
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt="[%(levelname)s %(asctime)s] %(filename)s: %(lineno)d  %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def validate_result(input_state_dict: dict[str, torch.Tensor], output_state_dict: dict[str, torch.Tensor]):
    input_state_dict = {k: v for k, v in input_state_dict.items() if isinstance(v, torch.Tensor)}
    output_state_dict = {k: v for k, v in output_state_dict.items() if isinstance(v, torch.Tensor)}

    input_key_set = set(input_state_dict.keys())
    output_key_set = set(output_state_dict.keys())

    if input_key_set != output_key_set:
        not_in_output_key_set = input_key_set - output_key_set
        not_in_input_key_set = output_key_set - input_key_set
        raise Exception(
            f"key set not equal, not in output key set: {not_in_output_key_set}, not in input key set: {not_in_input_key_set}"
        )

    not_equal_tensor = []
    for key in input_state_dict:
        if not torch.allclose(input_state_dict[key], output_state_dict[key]):
            not_equal_tensor.append(key)
    if len(not_equal_tensor) > 0:
        raise Exception(f"result is not valid, not equal tensors: {not_equal_tensor}")

    logger.info(f"all {len(input_key_set)} keys in state dict are equal")


def _get_available_cpu() -> int:
    avail_cpu = os.cpu_count()
    if os.path.isfile('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
        cpu_quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read().rstrip())
        if cpu_quota != -1 and os.path.isfile('/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
            cpu_period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read().rstrip())
            avail_cpu = int(cpu_quota / cpu_period)
            logger.info(f"get veturboio thread {avail_cpu} from cgroup info")
    return avail_cpu


class Pt2SafeTensorConverter:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        dry_run: bool,
        enable_to_valid_state_dict: bool,
        overwrite: bool,
        use_direct_io: bool,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.dry_run = dry_run
        self.enable_to_valid_state_dict = enable_to_valid_state_dict
        self.use_direct_io = use_direct_io
        if self.input_path.startswith("sfcs://"):
            try:
                self.input_file_size = veturboio.ops.sfcs_utils.sfcs_get_file_size(self.input_path)
            except BaseException as Exp:
                raise FileNotFoundError("can't get size of sfcs file", Exp)
        else:
            if not os.path.exists(self.input_path):
                raise Exception(f"file not exist: {self.input_path}")
            # convert to abs path
            if not os.path.isabs(self.input_path):
                self.input_path = os.path.abspath(self.input_path)
            self.input_file_size = os.path.getsize(self.input_path)
        if not self.input_path.endswith(".pt"):
            raise Exception("input file must end with .pt")

        if self.output_path is None:
            self.output_path = self.input_path.replace(".pt", ".safetensors")
        elif not self.output_path.startswith("sfcs://") and not os.path.isabs(self.output_path):
            self.output_path = os.path.abspath(self.output_path)
        if not self.output_path.endswith(".safetensors"):
            raise Exception("output file must end with .safetensors")

        if overwrite:
            if self.output_path.startswith("sfcs://"):
                raise Exception("overwrite flag cannot be set when using sfcs")
            if os.path.exists(self.output_path):
                logger.info(f"overwrite output file {self.output_path}")
                if not dry_run:
                    os.remove(self.output_path)
        elif not self.output_path.startswith("sfcs://") and os.path.exists(self.output_path):
            raise Exception(f"output file {self.output_path} already exists")

    def convert(self):
        logger.info(f"converting {self.input_path} to {self.output_path}")
        available_cpus = _get_available_cpu()
        ext_name = self.output_path.split(".")[-1]
        state_dict = {}
        if ext_name != "safetensors":
            raise ValueError("output file should be safetensors file")
        logger.info(f"start loading the pt file, the pt file has size of {self.input_file_size // 1000 // 1000}MB")
        start_time = datetime.now()
        if self.dry_run:
            logger.info("dry run finished for veturboio.load_pt_file")
        else:
            state_dict = veturboio.load(
                self.input_path, num_thread=available_cpus, use_direct_io=self.use_direct_io, enable_fast_mode=True
            )
        end_time = datetime.now()
        logger.info(f"finish loading the pt file with duration {end_time - start_time}")
        logger.info("start saving the safetensors file")
        start_time = datetime.now()
        if self.dry_run:
            logger.info("dry run finished for veturboio.save_safetensors_file")
        else:
            if self.enable_to_valid_state_dict:
                state_dict = to_valid_state_dict(state_dict)
            veturboio.save_file(state_dict, self.output_path, force_save_shared_tensor=True)
        end_time = datetime.now()
        logger.info(f"finish saving the safetensors file with duration {end_time - start_time}")

        del state_dict
        gc.collect()
        logger.info(f"gc finished")

    def validate(self):
        available_cpus = _get_available_cpu()
        logger.info(f"validating if {self.input_path} in equal to {self.output_path}")
        input_state_dict = veturboio.load(
            self.input_path, num_thread=available_cpus, use_direct_io=self.use_direct_io, enable_fast_mode=True
        )
        logger.info(f"{self.input_path} loaded")

        output_state_dict = veturboio.load(
            self.output_path, num_thread=available_cpus, use_direct_io=self.use_direct_io, enable_fast_mode=True
        )
        logger.info(f"{self.output_path} loaded")

        validate_result(input_state_dict, output_state_dict)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    add_handlers(logger)

    parser = argparse.ArgumentParser(description="converter used to convert .pt model to .safeTensor")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="indicate the path of .pt file, both posix path" "and sfcs prefix are supported",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        help="indicate the path of .safeTensor file, both "
        "posix path and sfcs prefix are supported."
        "will be placed into the same dir of the .pt "
        "file if left empty",
    )
    parser.add_argument("--dry-run", "-d", action="store_true", help="just dry run, not really convert")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the output file if it exists")
    parser.add_argument(
        "--enable-to-valid-state-dict",
        action="store_true",
        help="execute to_valid_state_dict function before save to .safetensors",
    )
    parser.add_argument("--validate-result", action="store_true", help="validate result", default=False)
    parser.add_argument("--use-direct-io", action="store_true", help="use direct io to load file", default=False)
    args = parser.parse_args()

    instance = Pt2SafeTensorConverter(
        args.input, args.output, args.dry_run, args.enable_to_valid_state_dict, args.overwrite, args.use_direct_io
    )
    try:
        instance.convert()
        if args.validate_result:
            instance.validate()
    except Exception as e:
        logger.error(f"convert failed.")
        traceback.print_exc()
        exit(1)
