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
import hashlib
import hmac
import json
import os
import secrets
import socket
from datetime import datetime
from enum import Enum
from time import sleep
from typing import Optional, Tuple

import numpy as np
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

try:
    import veturboio_ext

    CtrEncWrap = veturboio_ext.CtrEncWrap
    CtrDecWrap = veturboio_ext.CtrDecWrap
except ImportError:
    CtrEncWrap = None
    CtrDecWrap = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")


class SnapdConnection(HTTPConnection):
    def __init__(self, uds_path):
        super().__init__("localhost")
        self.uds_path = uds_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.uds_path)


class SnapdConnectionPool(HTTPConnectionPool):
    def __init__(self, uds_path):
        super().__init__("localhost")
        self.uds_path = uds_path

    def _new_conn(self):
        return SnapdConnection(self.uds_path)


class SnapdAdapter(HTTPAdapter):
    def __init__(self, uds_path):
        super().__init__()
        self.uds_path = uds_path

    def get_connection(self, url, proxies=None):
        return SnapdConnectionPool(self.uds_path)


class DataPipeClient:
    DATAPIPE_SOCKET_PATH = os.getenv('DATAPIPE_SOCKET_PATH', '/finetuned-model/datapipe.sock')
    PING_HEADER = {'X-Datapipe-Task-Type': 'ping'}
    ENCRYPT_HEADER = {
        'X-Datapipe-Task-Type': 'encrypt-key',
        'X-Encrypt-Caller-Pod': os.getenv('POD_NAME', ''),
        'X-TOS-Path': '',
    }
    SFCS_STS_HEADER = {'X-Datapipe-Task-Type': 'sfcs-sts'}
    KMS_STS_HEADER = {'X-Datapipe-Task-Type': 'kms-sts'}
    session = requests.Session()

    # Increment datapipe timeout to make it more robust to real scenarios
    def __init__(self, retry: int = 60, interval: float = 2) -> None:
        if not os.path.exists(self.DATAPIPE_SOCKET_PATH):
            raise RuntimeError(f'Datapipe socket {self.DATAPIPE_SOCKET_PATH} does not exist')

        self.url = 'http://localhost'
        self.session = requests.Session()
        self.session.mount(self.url, SnapdAdapter(self.DATAPIPE_SOCKET_PATH))
        self.retry = retry
        self.interval = interval
        resp = self._get_retry(self.PING_HEADER)
        if resp is None or resp['message'] != 'pong':
            raise RuntimeError(f'Ping Datapipe socket {self.DATAPIPE_SOCKET_PATH} failed')

    def _get_retry(self, headers: dict) -> Optional[dict]:
        re = 0
        while True:
            try:
                response = self.session.get(self.url, headers=headers)
                if response.status_code == 200:
                    return response.json()
                logger.warning(
                    f'call with {headers}, retry: {re}, code: {response.status_code}, body: {response.text}'
                )
            except Exception as e:
                logger.warning(f'call with {headers}, retry: {re}, raise exception: {e}')

            if re > self.retry:
                break
            sleep(self.interval)
            re += 1

        return None

    def get_data_key_iv(self, path: Optional[str] = None) -> Optional[dict]:
        header = self.ENCRYPT_HEADER.copy()
        if path:
            header['X-TOS-Path'] = path
        return self._get_retry(header)

    def get_sfcs_ak_sk_st(self) -> Optional[dict]:
        return self._get_retry(self.SFCS_STS_HEADER)

    def get_kms_ak_sk_st(self) -> Optional[dict]:
        return self._get_retry(self.KMS_STS_HEADER)


class KmsService:
    SERVICE = 'kms'

    def __init__(
        self,
        ak: str,
        sk: str,
        keyring_name: str,
        key_name: str,
        region: Optional[str] = None,
        host: Optional[str] = None,
        st: Optional[str] = None,
        uds_proxy: Optional[str] = None,
    ) -> None:
        self._ak = ak
        self._sk = sk
        self._st = st
        self._keyring_name = keyring_name
        self._key_name = key_name
        self._host = host or 'open.volcengineapi.com'
        self._region = region or 'cn-beijing'
        self._uds_proxy = uds_proxy

    @staticmethod
    def sign(key: bytes, msg: str):
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

    @staticmethod
    def getSignatureKey(key: str, dateStamp: str, regionName: str, serviceName: str):
        kDate = KmsService.sign(key.encode('utf-8'), dateStamp)
        kRegion = KmsService.sign(kDate, regionName)
        kService = KmsService.sign(kRegion, serviceName)
        kSigning = KmsService.sign(kService, 'request')
        return kSigning

    @staticmethod
    def formatParameters(parameters: dict):
        request_parameters_init = ''
        for key in sorted(parameters):
            request_parameters_init += key + '=' + parameters[key] + '&'
        request_parameters = request_parameters_init[:-1]
        return request_parameters

    @staticmethod
    def sigv4(
        ak: str,
        sk: str,
        host: str,
        region: str,
        srv: str,
        method: str,
        params: dict,
        payload: dict,
        st: Optional[str] = None,
        uds_proxy: Optional[str] = None,
    ) -> requests.Response:
        now = datetime.utcnow()
        current_date = now.strftime('%Y%m%dT%H%M%SZ')
        datestamp = now.strftime('%Y%m%d')
        cano_uri = '/'
        cano_query = KmsService.formatParameters(params)
        signed_headers = 'content-type;host;x-content-sha256;x-date'
        payload = json.dumps(payload)
        payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        cano_headers = (
            f'content-type:application/json\nhost:{host}\nx-content-sha256:{payload_hash}\nx-date:{current_date}\n'
        )
        cano_request = f'{method}\n{cano_uri}\n{cano_query}\n{cano_headers}\n{signed_headers}\n{payload_hash}'
        algorithm = 'HMAC-SHA256'
        cred_scope = f'{datestamp}/{region}/{srv}/request'
        string_to_sign = (
            f'{algorithm}\n{current_date}\n{cred_scope}\n' + hashlib.sha256(cano_request.encode('utf-8')).hexdigest()
        )
        signing_key = KmsService.getSignatureKey(sk, datestamp, region, srv)
        signature = hmac.new(signing_key, (string_to_sign).encode('utf-8'), hashlib.sha256).hexdigest()
        authorization_header = (
            f'{algorithm} Credential={ak}/{cred_scope}, SignedHeaders={signed_headers}, Signature={signature}'
        )
        headers = {
            'X-Date': current_date,
            'Authorization': authorization_header,
            'X-Content-Sha256': payload_hash,
            'Content-Type': 'application/json',
            'X-Amz-Date': '20180614T114308Z',
        }
        if st:
            headers['X-Security-Token'] = st
        request_url = f'https://{host}?{cano_query}'
        session = requests.Session()
        if uds_proxy:
            session.mount(f'https://{host}', SnapdAdapter(uds_proxy))
            headers['X-Datapipe-Task-Type'] = 'top'
        re = 0
        while True:
            try:
                resp = session.post(request_url, data=payload, headers=headers)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f'call kms with header: {headers}, return err:  {e}')
            if re > 3:
                break
            sleep(0.5)
            re += 1
        return resp

    def encrypt(self, pt_b64: str) -> str:
        params = {
            'Action': 'Encrypt',
            'Version': '2021-02-18',
            'KeyringName': self._keyring_name,
            'KeyName': self._key_name,
        }
        payload = {'Plaintext': pt_b64}
        js = KmsService.sigv4(
            self._ak,
            self._sk,
            self._host,
            self._region,
            self.SERVICE,
            'POST',
            params,
            payload,
            self._st,
            self._uds_proxy,
        )
        if 'Result' in js and 'CiphertextBlob' in js['Result']:
            return js['Result']['CiphertextBlob']
        raise RuntimeError(f'kms encrypt failed response: {js}')

    def decrypt(self, ct_b64: str) -> str:
        params = {
            'Action': 'Decrypt',
            'Version': '2021-02-18',
            'KeyringName': self._keyring_name,
            'KeyName': self._key_name,
        }
        payload = {'CiphertextBlob': ct_b64}
        js = KmsService.sigv4(
            self._ak,
            self._sk,
            self._host,
            self._region,
            self.SERVICE,
            'POST',
            params,
            payload,
            self._st,
            self._uds_proxy,
        )
        if 'Result' in js and 'Plaintext' in js['Result']:
            return js['Result']['Plaintext']
        raise RuntimeError(f'kms decrypt failed response: {js}')


class CipherMode(Enum):
    CTR_256 = 'CTR-256'
    CTR_128 = 'CTR-128'


class CipherInfo:
    ENV_KEY = 'VETURBOIO_KEY'
    ENV_IV = 'VETURBOIO_IV'
    ENV_KMS_HOST = 'VETURBOIO_KMS_HOST'
    ENV_KMS_REGION = 'VETURBOIO_KMS_REGION'
    ENV_KMS_AK = 'VETURBOIO_KMS_ACCESS_KEY'
    ENV_KMS_SK = 'VETURBOIO_KMS_SECRET_KEY'
    ENV_KMS_ST = 'VETURBOIO_KMS_SESSION_TOKEN'
    ENV_KMS_KEYRING = 'VETURBOIO_KMS_KEYRING_NAME'
    ENV_KMS_KEY = 'VETURBOIO_KMS_KEY_NAME'
    HEADER_SIZE = 262144
    MAGIC_NUMBER = b'Byte3ncryptM0del'

    def __init__(self, use_cipher: bool, header_bytes: Optional[bytes] = None, path: Optional[str] = None) -> None:
        self.use_cipher = use_cipher
        self.use_header = False
        self.mode = CipherMode.CTR_128
        self.key = np.frombuffer(b'\x00' * 16, dtype=np.byte)
        self.iv = np.frombuffer(b'\x00' * 16, dtype=np.byte)
        self.path = path
        if not use_cipher:
            return

        # case 1: get key and iv from file header part
        if (
            header_bytes is not None
            and len(header_bytes) == self.HEADER_SIZE
            and header_bytes[:16] == self.MAGIC_NUMBER
        ):
            # parse header to get key and iv
            self.use_header = True
            try:
                kms_srv = self.fetch_kms_client()
                first_zero = header_bytes.index(0)
                header_dict = json.loads(header_bytes[16:first_zero])
                self.mode = CipherMode(header_dict['mode'])
                key_b64 = kms_srv.decrypt(header_dict['meta_data_key'])
                iv_b64 = header_dict['iv']
                self.key, self.iv = self.convert_key_iv(key_b64, iv_b64)
                logger.info('get cipher info from file header successfully!')
                return
            except Exception as e:
                logger.warning(f'get cipher info from file header failed: {e}')

        # case 2: get key and iv from datapipe uds
        try:
            client = DataPipeClient()
            resp = client.get_data_key_iv(self.path)
            self.key, self.iv = self.convert_key_iv(resp['Key'], resp['IV'])
            logger.info('get cipher info from datapipe uds successfully!')
            return
        except Exception as e:
            logger.warning(f'get cipher info from datapipe uds failed: {e}')

        # case 3: get key and iv from env
        try:
            for e in [self.ENV_KEY, self.ENV_IV]:
                assert e in os.environ, f'env {e} not set'
            self.key, self.iv = self.convert_key_iv(os.getenv(self.ENV_KEY), os.getenv(self.ENV_IV))
            logger.info('get cipher info from env')
            return
        except Exception as e:
            logger.warning(f'get cipher info from env failed :{e}')

        # raise error
        logger.error('fail to get cipher info in all cases')
        raise RuntimeError('fail to get cipher info in all cases')

    @staticmethod
    def convert_key_iv(key_b64: str, iv_b64: str) -> Tuple[np.ndarray, np.ndarray]:
        key_b = base64.b64decode(key_b64, validate=True)
        iv_b = base64.b64decode(iv_b64, validate=True)
        if (len(key_b) != 16 and len(key_b) != 32) or len(iv_b) != 16:
            raise Exception(f'length of key {len(key_b)} or iv {len(iv_b)} is not valid')
        key = np.frombuffer(key_b, dtype=np.byte)
        iv = np.frombuffer(iv_b, dtype=np.byte)
        return key, iv

    def fetch_kms_client(self) -> KmsService:
        kms_host = os.getenv(self.ENV_KMS_HOST)
        region = os.getenv(self.ENV_KMS_REGION)
        ak = os.getenv(self.ENV_KMS_AK)
        sk = os.getenv(self.ENV_KMS_SK)
        st = os.getenv(self.ENV_KMS_ST)
        keyring_name = os.getenv(self.ENV_KMS_KEYRING)
        key_name = os.getenv(self.ENV_KMS_KEY)
        uds_proxy = None
        # try to fetch kms credential from datapipe
        if os.path.exists(DataPipeClient.DATAPIPE_SOCKET_PATH):
            try:
                client = DataPipeClient()
                uds_proxy = DataPipeClient.DATAPIPE_SOCKET_PATH
                resp = client.get_kms_ak_sk_st()
                ak = resp['Cred']['AccessKeyId']
                sk = resp['Cred']['SecretAccessKey']
                st = resp['Cred']['SessionToken']
                logger.info('get kms ak/sk/st from datapipe successfully!')
            except Exception as e:
                logger.warning(f'get kms ak/sk/st from datapipe failed: {e}')

        for var in [ak, sk, keyring_name, key_name]:
            assert var is not None, 'required kms info not set'
        return KmsService(ak, sk, keyring_name, key_name, region, kms_host, st, uds_proxy)

    def to_header_bytes(self) -> bytearray:
        kms_srv = self.fetch_kms_client()
        header_dict = {
            'mode': self.mode.value,
            'iv': base64.b64encode(self.iv.data).decode('utf-8'),
            'meta_data_key': kms_srv.encrypt(base64.b64encode(self.key).decode('utf-8')),
            'file_timestamp': int(datetime.utcnow().timestamp()),
        }
        header_json = json.dumps(header_dict)
        header_bytes = bytearray(self.HEADER_SIZE)
        header_bytes[:16] = self.MAGIC_NUMBER
        header_bytes[16 : 16 + len(header_json)] = header_json.encode('utf-8')
        return header_bytes


def create_cipher_with_header(mode: CipherMode, path: str) -> CipherInfo:
    c = CipherInfo(False, None, path)
    c.use_cipher = True
    c.use_header = True
    c.mode = mode
    if c.mode == CipherMode.CTR_256:
        key_bytes = secrets.token_bytes(32)
    else:
        key_bytes = secrets.token_bytes(16)
    iv_bytes = secrets.token_bytes(16)
    c.key = np.frombuffer(key_bytes, dtype=np.byte)
    c.iv = np.frombuffer(iv_bytes, dtype=np.byte)
    return c


def encrypt(cipher_info: CipherInfo, pt: np.ndarray, ct: np.ndarray, offset: int):
    # note: dtype of pt and ct should be np.uint8
    if not cipher_info.use_cipher:
        logger.warning('cipher.encrypt: use_cipher False, skip')
        return
    enc = CtrEncWrap(cipher_info.mode.value, cipher_info.key, cipher_info.iv, offset)
    ret = enc.encrypt_update(pt, ct)
    if not ret:
        logger.error('cipher.encrypt: failed')


def decrypt(cipher_info: CipherInfo, ct: np.ndarray, pt: np.ndarray, offset: int):
    # note: dtype of pt and ct should be np.uint8
    if not cipher_info.use_cipher:
        logger.warning('cipher.decrypt: use_cipher False, skip')
        return
    dec = CtrDecWrap(cipher_info.mode.value, cipher_info.key, cipher_info.iv, offset)
    ret = dec.decrypt_update(ct, pt)
    if not ret:
        logger.error('cipher.decrypt: failed')
