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
import os
import time
from functools import lru_cache

import numpy as np
import torch

import veturboio


def human_read_to_byte(size):
    factors = {
        'B': 1,
        'KB': 1024,
        'MB': 1048576,
        'GB': 1073741824,
        'TB': 1099511627776,
        'PB': 1125899906842624,
        'EB': 1152921504606846976,
        'ZB': 1180591620717411303424,
        'YB': 1208925819614629174706176,
    }
    if size[-2:] in factors:
        return factors[size[-2:]] * int(size[:-2])
    elif size[-1:] in factors:
        return int(size[:-1])
    else:
        return int(size)


def parse_args():
    parser = argparse.ArgumentParser(
        description='benchmark veturboio, notice to clear page cache manually when benchmarking for existing file'
    )
    parser.add_argument(
        '--begin',
        default='1048576',
        dest='begin',
        help='specify the minimum file size to benchmark in bytes or in format like xxKB/MB/GB',
    )
    parser.add_argument(
        '--end',
        default='1048576',
        dest='end',
        help='specify the maximum file size to benchmark in bytes or in format like xxKB/MB/GB',
    )
    parser.add_argument('--base_dir', dest='base_dir', help='specify the the base dir of files to be benchmarked')
    parser.add_argument('--fs_name', default='local_fs', help='file system name that would be displayed in the result')
    parser.add_argument('--gen_data', default=False, action=argparse.BooleanOptionalAction, dest='gen_data')
    parser.add_argument(
        '--map_location', default='cpu', dest='map_location', help='map location of tensor to be loaded'
    )
    parser.add_argument('--use_pinmem', default=False, action=argparse.BooleanOptionalAction, dest='use_pinmem')
    parser.add_argument(
        '--load_mode', default='veturboio', dest='load_mode', help='load modes specified, seperated by comma'
    )

    args = parser.parse_args()
    return args


def print_header(load_modes):
    mode_list = list(map(lambda mode: f"{mode}{' load_time(s)' + ' ':<25}", load_modes))
    print(f"{'fs_name' + ' ':<10} {'tensor_size' + ' ':<15}", ' '.join(mode_list))


def print_load_time(fs_name, tensor_size, load_times):
    load_times = list(map(lambda load_time: f"{load_time}{' ':<30}", load_times))
    print(f"{fs_name:<10} {str(tensor_size):<15}", ' '.join(load_times))


def sfcs_env():
    os.environ['SFCS_FSNAME'] = 'byted-cpu-sfcs'
    os.environ['SFCS_REGION'] = 'cn-beijing'
    os.environ['SFCS_ACCESS_KEY'] = os.environ['CI_SFCS_AK']
    os.environ['SFCS_SECRET_KEY'] = os.environ['CI_SFCS_SK']
    os.environ['SFCS_AUTHENTICATION_SERVICE_NAME'] = 'cfs'
    os.environ['SFCS_NS_ID'] = '18014398509481988'
    os.environ['SFCS_UFS_PATH'] = 'tos://yinzq-bucket/'
    os.environ['SFCS_MULTI_NIC_WHITELIST'] = 'eth0'
    os.environ['SFCS_NETWORK_SEGMENT'] = '172.31.128.0/17'
    os.environ['SFCS_NAMENODE_ENDPOINT_ADDRESS'] = '100.67.19.231'
    os.environ['SFCS_LOG_SEVERITY'] = 'ERROR'


def main():
    args = parse_args()
    if args.base_dir.startswith('sfcs://'):
        sfcs_env()
    load_modes = args.load_mode.split(',')
    # warmup GPU otherwise the first case would be slow
    device = torch.device(args.map_location)
    if device.type == "cuda":
        file_path = os.path.join(args.base_dir if args.base_dir else "", 'warmup.safetensors')
        tensors = {"weight": torch.randn(10)}
        veturboio.save_file(tensors, file_path)
        veturboio.load(file_path, map_location=args.map_location, use_pinmem=args.use_pinmem)
    print_header(load_modes)
    tensor_size = human_read_to_byte(args.begin)
    end_size = human_read_to_byte(args.end)
    while tensor_size <= end_size:
        if args.gen_data:
            numel = tensor_size // np.dtype(float).itemsize * 2
            tensors = {"weight": torch.randn(numel)}
        load_times = []
        for mode in load_modes:
            if mode == 'veturboio':
                file_path = os.path.join(args.base_dir if args.base_dir else "", f'{tensor_size}.safetensors')
                if args.gen_data:
                    veturboio.save_file(tensors, file_path)

                start = time.time()
                loaded_tensor = veturboio.load(file_path, map_location=args.map_location, use_pinmem=args.use_pinmem)
            if mode == 'torch':
                file_path = os.path.join(args.base_dir if args.base_dir else "", f'{tensor_size}.pt')
                if args.gen_data:
                    veturboio.save_pt(tensors, file_path)

                start = time.time()

                loaded_tensor = veturboio.load(file_path, map_location=args.map_location)
            end = time.time()
            load_times.append("%.2f" % (end - start))

            if device.type == "cuda":
                del loaded_tensor
                torch.cuda.empty_cache()

        print_load_time(args.fs_name, tensor_size, load_times)
        tensor_size = tensor_size * 2


if __name__ == '__main__':
    main()
