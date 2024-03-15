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
import http.server
import json
import os
import socketserver
import tempfile
import threading
from datetime import datetime, timedelta
from time import sleep
from unittest import TestCase

import numpy as np

from veturboio.ops.cipher import CipherInfo, DataPipeClient
from veturboio.ops.sfcs_utils import (
    SFCS_OPT_ENV_LIST,
    SFCS_PROPERTIES,
    SFCS_REQ_ENV_LIST,
    credentials_helper,
    init_sfcs_conf,
)


class UnixSocketHttpServer(socketserver.UnixStreamServer):
    def get_request(self):
        request, client_address = super().get_request()
        return (request, ["local", 0])


class DatapipeHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        action = self.headers.get('X-Datapipe-Task-Type')
        if action == 'top':
            # mock kms response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            res = {'Result': {'Plaintext': base64.b64encode(b'abcdefgh87654321').decode('ascii')}}
            self.wfile.write(bytes(json.dumps(res), encoding='ascii'))
            return
        self.send_response(400)
        self.end_headers()
        return

    def do_GET(self):
        action = self.headers.get('X-Datapipe-Task-Type')
        if action == 'ping':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'message': 'pong'}), encoding='ascii'))
            return
        if action == 'encrypt-key':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps({'Key': 'YWJjZGVmZ2gxMjM0NTY3OA==', 'IV': 'MTIzNDU2Nzg4NzY1NDMyMQ=='}), encoding='ascii'
                )
            )
            return
        if action == 'sfcs-sts':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            date_now = datetime.now()
            date_exp = date_now + timedelta(seconds=4)
            res = {
                'Cred': {
                    'CurrentTime': date_now.isoformat(),
                    'ExpiredTime': date_exp.isoformat(),
                    'AccessKeyId': 'AKTPODg0MzV**2ZDcxMDg',
                    'SecretAccessKey': 'TVRNNVlqRmxPR1**mRoTkdWbE1ESQ==',
                    'SessionToken': 'STSeyJBY2NvdW50SW**kXXXXXXX',  # fake SessionToken real one is longer
                },
                'SfcsNameNodeAddress': '100.67.19.231',
            }
            self.wfile.write(bytes(json.dumps(res), encoding='ascii'))
            return
        if action == 'kms-sts':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            res = {
                'Cred': {
                    'AccessKeyId': os.environ['CI_VENDOR_AK'],
                    'SecretAccessKey': os.environ['CI_VENDOR_AK'],
                    'SessionToken': '',
                },
            }
            self.wfile.write(bytes(json.dumps(res), encoding='ascii'))
            return
        self.send_response(400)
        self.end_headers()
        return


class TestCipherInfo(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sock_dir = tempfile.TemporaryDirectory()
        cls.server_address = os.path.join(cls.sock_dir.name, 'datapipe.sock')
        cls.server = UnixSocketHttpServer(cls.server_address, DatapipeHandler, bind_and_activate=True)

        def run():
            cls.server.serve_forever()

        cls.thread = threading.Thread(target=run)
        cls.thread.start()
        cls.target_key = np.frombuffer(b'abcdefgh12345678', dtype=np.byte)
        cls.target_key_2 = np.frombuffer(b'abcdefgh87654321', dtype=np.byte)
        cls.target_iv = np.frombuffer(b'1234567887654321', dtype=np.byte)

    def test_fetch_from_file_header(self):
        os.environ.pop('VETURBOIO_KEY', None)
        os.environ.pop('VETURBOIO_IV', None)
        DataPipeClient.DATAPIPE_SOCKET_PATH = '/path/not/exist'

        header_dict = {
            'mode': 'CTR-128',
            'iv': 'MTIzNDU2Nzg4NzY1NDMyMQ==',
            'meta_data_key': 'bl2htKYLQ2+CjyyJ84Q3twAA9ZpCbFxwznRb0NkR9zGGRp1RK5Mb9u8NNOiahY+0yVrxNw3IVQ9Wgn6PDscw77Cb3eImjVn14hNBJRlwtSyQ7tRZLOsZBEHv5cWwDQ==',
        }
        header_bytes = bytearray(256 * 1024)
        header_str = 'Byte3ncryptM0del' + json.dumps(header_dict)
        header_bytes[: len(header_str)] = header_str.encode('utf-8')

        # case1: get kms cred from env
        ENV_KMS_HOST = 'VETURBOIO_KMS_HOST'
        ENV_KMS_REGION = 'VETURBOIO_KMS_REGION'
        ENV_KMS_AK = 'VETURBOIO_KMS_ACCESS_KEY'
        ENV_KMS_SK = 'VETURBOIO_KMS_SECRET_KEY'
        ENV_KMS_KEYRING = 'VETURBOIO_KMS_KEYRING_NAME'
        ENV_KMS_KEY = 'VETURBOIO_KMS_KEY_NAME'
        os.environ[ENV_KMS_HOST] = 'open.volcengineapi.com'
        os.environ[ENV_KMS_REGION] = 'cn-beijing'
        os.environ[ENV_KMS_AK] = os.environ['CI_VENDOR_AK']
        os.environ[ENV_KMS_SK] = os.environ['CI_VENDOR_SK']
        os.environ[ENV_KMS_KEYRING] = 'datapipe_keyring'
        os.environ[ENV_KMS_KEY] = 'datapipe_key_ml_maas'
        info = CipherInfo(True, header_bytes)
        self.assertTrue(info.use_cipher)
        self.assertTrue(info.use_header)
        self.assertTrue(np.array_equal(info.key, self.target_key))
        self.assertTrue(np.array_equal(info.iv, self.target_iv))

        # case2: get kms cred from datapipe and access kms with datapipe proxy
        os.environ.pop(ENV_KMS_HOST, None)
        os.environ.pop(ENV_KMS_REGION, None)
        os.environ.pop(ENV_KMS_AK, None)
        os.environ.pop(ENV_KMS_SK, None)
        DataPipeClient.DATAPIPE_SOCKET_PATH = self.server_address
        info = CipherInfo(True, header_bytes)
        info = CipherInfo(True, header_bytes)
        self.assertTrue(info.use_cipher)
        self.assertTrue(info.use_header)
        self.assertTrue(np.array_equal(info.key, self.target_key_2))
        self.assertTrue(np.array_equal(info.iv, self.target_iv))

    def test_fetch_from_datapipe(self):
        DataPipeClient.DATAPIPE_SOCKET_PATH = self.server_address
        info = CipherInfo(True)
        self.assertTrue(info.use_cipher)
        self.assertTrue(np.array_equal(info.key, self.target_key))
        self.assertTrue(np.array_equal(info.iv, self.target_iv))

    def test_fetch_from_env(self):
        DataPipeClient.DATAPIPE_SOCKET_PATH = '/path/not/exist'
        os.environ['VETURBOIO_KEY'] = base64.b64encode(b'abcdefgh12345678').decode('ascii')
        os.environ['VETURBOIO_IV'] = base64.b64encode(b'1234567887654321').decode('ascii')
        info = CipherInfo(True)
        self.assertTrue(info.use_cipher)
        self.assertTrue(np.array_equal(info.key, self.target_key))
        self.assertTrue(np.array_equal(info.iv, self.target_iv))

    def test_fallback(self):
        DataPipeClient.DATAPIPE_SOCKET_PATH = '/path/not/exist'
        os.environ['VETURBOIO_KEY'] = base64.b64encode(b'abcdefgh12').decode('ascii')
        os.environ['VETURBOIO_IV'] = base64.b64encode(b'1234567887').decode('ascii')
        info = CipherInfo(True)
        self.assertFalse(info.use_cipher)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop('VETURBOIO_KEY', None)
        os.environ.pop('VETURBOIO_IV', None)
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.sock_dir.cleanup()


class TestCredentials(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sock_dir = tempfile.TemporaryDirectory()
        cls.server_address = os.path.join(cls.sock_dir.name, 'datapipe.sock')
        cls.server = UnixSocketHttpServer(cls.server_address, DatapipeHandler, bind_and_activate=True)

        def run():
            cls.server.serve_forever()

        cls.thread = threading.Thread(target=run)
        cls.thread.start()

    def test_sfcs_sts(self):
        DataPipeClient.DATAPIPE_SOCKET_PATH = self.server_address
        client = DataPipeClient()
        cred = client.get_sfcs_ak_sk_st()
        self.assertIsNotNone(cred)
        self.assertEqual(cred['SfcsNameNodeAddress'], '100.67.19.231')
        cred = cred['Cred']
        self.assertEqual(cred['AccessKeyId'], 'AKTPODg0MzV**2ZDcxMDg')
        self.assertEqual(cred['SecretAccessKey'], 'TVRNNVlqRmxPR1**mRoTkdWbE1ESQ==')
        self.assertEqual(cred['SessionToken'], 'STSeyJBY2NvdW50SW**kXXXXXXX')

    def test_sfcs_conf(self):
        # case 1: a xml file already exists, do nothing
        with tempfile.NamedTemporaryFile() as sfcs_conf:
            os.environ['LIBCFS_CONF'] = sfcs_conf.name
            init_sfcs_conf()
            self.assertFalse(credentials_helper.running)

        for e in SFCS_REQ_ENV_LIST:
            os.environ[e] = 'test-value'

        # case 2: env SFCS_ACCESS_KEY and SFCS_SECRET_KEY and SFCS_NAMENODE_ENDPOINT_ADDRESS exists
        with tempfile.TemporaryDirectory() as conf_dir:
            conf_path = os.path.join(conf_dir, 'libcfs.xml')
            os.environ['LIBCFS_CONF'] = conf_path
            os.environ['SFCS_ACCESS_KEY'] = 'AKTPODg0MzV**2ZDcxMDg'
            os.environ['SFCS_SECRET_KEY'] = 'TVRNNVlqRmxPR1**mRoTkdWbE1ESQ=='
            os.environ['SFCS_NAMENODE_ENDPOINT_ADDRESS'] = '100.67.19.231'
            init_sfcs_conf()
            self.assertEqual(SFCS_PROPERTIES['cfs.access.key'], 'AKTPODg0MzV**2ZDcxMDg')
            self.assertEqual(SFCS_PROPERTIES['cfs.secret.key'], 'TVRNNVlqRmxPR1**mRoTkdWbE1ESQ==')
            self.assertEqual(SFCS_PROPERTIES['cfs.namenode.endpoint.address.test-value'], '100.67.19.231')
            self.assertFalse(credentials_helper.running)
            self.assertTrue(os.path.exists(conf_path))

        # case 3: use datapipe socket to get and refresh ak, sk, st and namenode_ip
        DataPipeClient.DATAPIPE_SOCKET_PATH = self.server_address
        with tempfile.TemporaryDirectory() as conf_dir:
            conf_path = os.path.join(conf_dir, 'libcfs.xml')
            os.environ['LIBCFS_CONF'] = conf_path
            os.environ.pop('SFCS_ACCESS_KEY', None)
            os.environ.pop('SFCS_SECRET_KEY', None)
            os.environ.pop('SFCS_NAMENODE_ENDPOINT_ADDRESS', None)
            SFCS_PROPERTIES.pop('cfs.access.key')
            SFCS_PROPERTIES.pop('cfs.secret.key')
            SFCS_PROPERTIES.pop('cfs.namenode.endpoint.address.test-value')
            init_sfcs_conf()
            self.assertEqual(SFCS_PROPERTIES['cfs.access.key'], 'AKTPODg0MzV**2ZDcxMDg')
            self.assertEqual(SFCS_PROPERTIES['cfs.secret.key'], 'TVRNNVlqRmxPR1**mRoTkdWbE1ESQ==')
            self.assertEqual(SFCS_PROPERTIES['cfs.namenode.endpoint.address.test-value'], '100.67.19.231')
            self.assertEqual(SFCS_PROPERTIES['cfs.security.token'], 'STSeyJBY2NvdW50SW**kXXXXXXX')
            self.assertTrue(credentials_helper.running)
            self.assertTrue(os.path.exists(conf_path))
            t1 = credentials_helper.current_time
            sleep(3)
            t2 = credentials_helper.current_time
            self.assertTrue(t1 < t2)
            credentials_helper.stop()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop('LIBCFS_CONF', None)
        for e in SFCS_REQ_ENV_LIST:
            os.environ.pop(e, None)
        for e in SFCS_OPT_ENV_LIST:
            os.environ.pop(e, None)
        SFCS_PROPERTIES.pop('cfs.security.token', None)
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.sock_dir.cleanup()
