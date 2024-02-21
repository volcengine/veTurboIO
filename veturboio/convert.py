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

import torch

from veturboio import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, required=True)
parser.add_argument("--output", "-o", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"convert {args.input} to {args.output}")
    ext_name = args.output.split(".")[-1]
    if ext_name != "safetensors":
        raise ValueError("output file should be safetensors file")
    state_dict = torch.load(args.input)
    save_file(state_dict, args.output, force_save_shared_tensor=True)
