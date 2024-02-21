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

import base64
import os
import threading
import urllib.parse
from datetime import datetime, timezone
from time import sleep
from typing import Optional, Tuple

import numpy as np
import requests_unixsocket
from loguru import logger


class DataPipeClient:
    DATAPIPE_SOCKET_PATH = os.getenv('DATAPIPE_SOCKET_PATH', '/finetune/data/datapipe.sock')
    ENCRYPT_HEADER = {'X-Datapipe-Task-Type': 'encrypt-key'}
    SFCS_STS_HEADER = {'X-Datapipe-Task-Type': 'sfcs-sts'}

    def __init__(self, retry: int = 3, interval: float = 0.5) -> None:
        if os.path.exists(self.DATAPIPE_SOCKET_PATH):
            self.url = 'http+unix://' + urllib.parse.quote(self.DATAPIPE_SOCKET_PATH, safe='')
            self.session = requests_unixsocket.Session()
            self.retry = retry
            self.interval = interval
        else:
            self.url = None
            self.session = None

    def get_data_key_iv(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.session:
            logger.warning('Datapipe client initialization failed')
            return None, None

        re = 0
        while True:
            try:
                response = self.session.get(self.url, headers=self.ENCRYPT_HEADER)
                if response.status_code == 200:
                    res = response.json()
                    return res['Key'], res['IV']
            except Exception as e:
                logger.warning(e)

            if re > self.retry:
                break
            sleep(self.interval)
            re += 1

        return None, None

    def get_sfcs_ak_sk_st(self) -> Optional[dict]:
        if not self.session:
            logger.warning('Datapipe client initialization failed')
            return None

        re = 0
        while True:
            try:
                response = self.session.get(self.url, headers=self.SFCS_STS_HEADER)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                logger.warning(e)

            if re > self.retry:
                break
            sleep(self.interval)
            re += 1

        return None


class CipherInfo:
    ENV_KEY = 'VETUROIO_KEY'
    ENV_IV = 'VETUROIO_IV'

    def __init__(self, use_cipher: bool) -> None:
        if use_cipher:
            # first try to get key and iv from datapipe
            client = DataPipeClient()
            if client.session:
                try:
                    key_b64, iv_b64 = client.get_data_key_iv()
                    self.key, self.iv = self.convert_key_iv(key_b64, iv_b64)
                    self.use_cipher = True
                    logger.info('get cipher info from datapipe socket')
                    return
                except Exception as e:
                    logger.warning(e)

            # then try to get key and iv from env
            env_key = os.getenv(self.ENV_KEY)
            env_iv = os.getenv(self.ENV_IV)
            if env_key and env_iv:
                try:
                    self.key, self.iv = self.convert_key_iv(env_key, env_iv)
                    self.use_cipher = True
                    logger.info('get cipher info from env')
                    return
                except Exception as e:
                    logger.warning(e)
            logger.warning('fail to get key and iv, fallback to no cipher')

        self.use_cipher = False
        self.key = np.frombuffer(b'\x00' * 16, dtype=np.byte)
        self.iv = np.frombuffer(b'\x00' * 16, dtype=np.byte)

    @staticmethod
    def convert_key_iv(key_b64: str, iv_b64: str) -> Tuple[np.ndarray, np.ndarray]:
        key_b = base64.b64decode(key_b64, validate=True)
        iv_b = base64.b64decode(iv_b64, validate=True)
        if len(key_b) != 16 or len(iv_b) != 16:
            raise Exception('length of key or iv is not 16')
        key = np.frombuffer(key_b, dtype=np.byte)
        iv = np.frombuffer(iv_b, dtype=np.byte)
        return key, iv
