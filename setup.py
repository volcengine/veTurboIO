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
import sys

import requests
import setuptools
import torch
from pkg_resources import parse_version
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths

# initialize variables for compilation
IS_LINUX = platform.system() == "Linux"
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

this_dir = os.path.dirname(os.path.abspath(__file__))


def get_option():
    if os.getenv("NPU_EXTENSION_ENABLED", "0") == "1":
        sys.argv.append("--npu_ext")
    elif "--cuda_ext" not in sys.argv and "--npu_ext" not in sys.argv and "--cpu_ext" not in sys.argv:
        print(
            '''No known extension specified, default to use --cuda_ext. Currently supported:
            --cuda_ext
            --npu_ext
            --cpu_ext'''
        )
        sys.argv.append("--cuda_ext")


def get_version():
    import importlib.util

    spec = importlib.util.spec_from_file_location("version", os.path.join("veturboio", "version.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    if "--cpu_ext" in sys.argv:
        return m.__version__ + "+cpu"
    elif "--npu_ext" in sys.argv:
        return m.__version__ + "+npu"
    else:
        return m.__version__


def make_relative_rpath(path):
    if IS_DARWIN:
        return '-Wl,-rpath,@loader_path/' + path
    elif IS_WINDOWS:
        return ''
    else:
        return '-Wl,-rpath,$ORIGIN/' + path


def get_veturboio_extension():
    get_option()
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
    extra_compile_args = {'cxx': ['-fvisibility=hidden'], 'nvcc': ['-O3']}

    if parse_version(torch.__version__) <= parse_version('1.12.1'):
        extra_compile_args['cxx'].append('-std=c++14')
    else:
        extra_compile_args['cxx'].append('-std=c++17')

    name = "veturboio_ext"

    sources = [
        "veturboio/ops/csrc/pybind.cpp",
        "veturboio/ops/csrc/posix.cpp",
        "veturboio/ops/csrc/sfcs.cpp",
        "veturboio/ops/csrc/io_helper_cpu_common.cpp",
        "veturboio/ops/csrc/cipher.cpp",
    ]

    include_dirs = include_paths()
    include_dirs.append("veturboio/ops/csrc/include")

    torch_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    library_dirs = [torch_dir]
    library_dirs.append("veturboio/ops/csrc/lib")

    libraries = ["cloudfs", ":libfastcrypto_gpu.so.0.3"]

    extra_link_args = [make_relative_rpath("veturboio/ops/csrc/lib")]

    # Refer to: https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py#L918
    # In torch 2.0, this flag is False, and the *.so lib set this flag as False when building.
    # In newer torch, this flag is True, to keep compatibility with *.so lib, we set it False
    # to generate g++ flags '-D_GLIBCXX_USE_CXX11_ABI=0' when building veturboio_ext, otherwise
    # some 'undefine symbol' error of std::string will be thrown.
    torch._C._GLIBCXX_USE_CXX11_ABI = False

    if "--cuda_ext" in sys.argv:
        sys.argv.remove("--cuda_ext")

        extra_compile_args['nvcc'].append('-O3')

        sources.append("veturboio/ops/csrc/io_helper.cu")

        define_macros.append(("USE_CUDA", "1"))

        from torch.utils.cpp_extension import CUDAExtension

        return CUDAExtension(
            name=name,
            sources=sources,
            define_macros=define_macros,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    else:
        extra_compile_args['cxx'].append('-O3')

        libraries.append("torch_cpu")
        libraries.append("torch_python")

        extra_link_args.append(f"-Wl,--rpath={torch_dir},--enable-new-dtags")

        if "--npu_ext" in sys.argv:
            sys.argv.remove("--npu_ext")

            sources.append("veturboio/ops/csrc/io_helper_npu.cpp")
            define_macros.append(("USE_NPU", "1"))

            return Extension(
                name=name,
                sources=sources,
                define_macros=define_macros,
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        elif "--cpu_ext" in sys.argv:
            sys.argv.remove("--cpu_ext")

            sources.append("veturboio/ops/csrc/io_helper_cpu.cpp")

            return Extension(
                name=name,
                sources=sources,
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
    user_options = [('src=', 's', 'source url of libcloudfs.so'), ('dst=', 'd', 'dest filepath of libcloudfs.so')]

    def initialize_options(self):
        from veturboio.utils.load_veturboio_ext import LIBCFS_DEFAULT_PATH, LIBCFS_DEFAULT_URL

        self.src = LIBCFS_DEFAULT_URL
        self.dst = LIBCFS_DEFAULT_PATH

    def finalize_options(self):
        pass

    def run(self):
        print(f"download libcloudfs.so from {self.src}, save to {self.dst}")
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
        "netifaces",
        "loguru",
        "requests-unixsocket",
        "requests",
    ],
    include_package_data=True,
    cmdclass={"get_libcfs": GetLibCfsCommand, "build_ext": BuildExtension},
    dependency_links=['https://mirrors.ivolces.com/pypi/'],
)
