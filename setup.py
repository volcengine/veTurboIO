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
import platform

import requests
import setuptools
import torch
from pkg_resources import parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# initialize variables for compilation
IS_LINUX = platform.system() == "Linux"
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"


def get_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location("version", os.path.join("veturboio", "version.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    return m.__version__


def make_relative_rpath(path):
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        return ''
    else:
        return '-Wl,-rpath,$ORIGIN/' + path


def get_veturboio_extension():
    # prevent ninja from using too many resources
    try:
        import psutil

        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault("MAX_JOBS", str(cpu_use))
    # os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6"

    define_macros = []

    # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
    # required key passed to PyTorch. Even if there is no flag passed
    # to cxx, users also need to pass an empty list to PyTorch.
    # Since PyTorch1.8.0, it has a default value so users do not need
    # to pass an empty list anymore.
    # More details at https://github.com/pytorch/pytorch/pull/45956
    extra_compile_args = {'cxx': [], 'nvcc': ['-O3']}

    if parse_version(torch.__version__) <= parse_version('1.12.1'):
        extra_compile_args['cxx'] = ['-std=c++14']
    else:
        extra_compile_args['cxx'] = ['-std=c++17']

    include_dirs = ["veturboio/ops/csrc/include"]
    library_dirs = ["veturboio/ops/csrc/lib"]
    libraries = ["cfs", "fastcrypto"]
    extra_link_args = [make_relative_rpath("veturboio/ops/csrc/lib")]

    return CUDAExtension(
        name="veturboio_ext",
        sources=[
            "veturboio/ops/csrc/pybind.cpp",
            "veturboio/ops/csrc/load_utils.cpp",
            "veturboio/ops/csrc/sfcs.cpp",
            "veturboio/ops/csrc/io_helper.cu",
        ],
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


class GetLibCfsCommand(setuptools.Command):
    """get libcfs from url"""

    description = 'get libcfs from url'
    user_options = [('src=', 's', 'source url of libcfs.so'), ('dst=', 'd', 'dest filepath of libcfs.so')]

    def initialize_options(self):
        from veturboio.utils.load_veturboio_ext import LIBCFS_DEFAULT_PATH, LIBCFS_DEFAULT_URL

        self.src = LIBCFS_DEFAULT_URL
        self.dst = LIBCFS_DEFAULT_PATH

    def finalize_options(self):
        pass

    def run(self):
        print(f"download libcfs.so from {self.src}, save to {self.dst}")
        r = requests.get(self.src, timeout=60)
        with open(self.dst, 'wb') as f:
            f.write(r.content)


setup(
    name="veturboio",
    version=get_version(),
    description="Effcient PyTorch IO libraray on Volcanic Engine",
    author="AML Team",
    ext_modules=[get_veturboio_extension()],
    packages=find_packages(exclude=("veturboio.ops.csrc.common.sfcs.lib")),
    install_requires=[
        "safetensors",
        "numpy",
        "loguru",
        "requests-unixsocket",
        "requests",
    ],
    include_package_data=True,
    cmdclass={"get_libcfs": GetLibCfsCommand, "build_ext": BuildExtension},
)
