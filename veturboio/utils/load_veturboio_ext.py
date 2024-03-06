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

from loguru import logger

LIBCFS_DEFAULT_URL = "https://veturbo-cn-beijing.tos-cn-beijing.volces.com/veturboio/libcfs/libcfs.so"
LIBCFS_DEFAULT_PATH = "/usr/lib/libcfs.so"


def load_libcfs():
    libcfs_path = os.getenv("LIBCFS_PATH", LIBCFS_DEFAULT_PATH)
    if not os.path.isfile(libcfs_path):
        # libcfs_path not exist, download from url
        import requests

        libcfs_url = os.getenv("LIBCFS_URL", LIBCFS_DEFAULT_URL)
        logger.info(f"download libcfs.so from {libcfs_url}, save to {libcfs_path}")
        r = requests.get(libcfs_url, timeout=60)
        with open(libcfs_path, 'wb') as f:
            f.write(r.content)


def load_veturboio_ext():
    load_libcfs()
    import veturboio_ext

    return veturboio_ext
