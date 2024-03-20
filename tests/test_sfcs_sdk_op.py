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
import tempfile
import unittest
from copy import deepcopy
from unittest import TestCase

import numpy as np
import torch

import veturboio
import veturboio.ops.sfcs_utils as sfcs_utils


def init_sfcs_env():
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


class TestSFCS(TestCase):
    @classmethod
    def setUpClass(cls):
        init_sfcs_env()

    def _run_pipeline(self):
        filepath = "/data.bin"
        filesize = 1024 * 1024

        first_path = os.path.abspath(filepath).split("/")[1]
        sfcs_conf = os.path.join(os.getcwd(), first_path + '.xml')
        if os.path.exists(sfcs_conf):
            os.remove(sfcs_conf)
        sfcs_utils.init_sfcs_conf(filepath)

        sfcs_utils.sfcs_delete_file(filepath)

        arr_0 = np.empty([filesize], dtype=np.byte)
        length = sfcs_utils.sfcs_write_file(filepath, arr_0, filesize)
        self.assertEqual(length, filesize)

        size = sfcs_utils.sfcs_get_file_size(filepath)
        self.assertEqual(size, filesize)

        arr_1 = np.empty([filesize], dtype=np.byte)
        length = sfcs_utils.sfcs_read_file(filepath, arr_1, filesize, 0)
        self.assertEqual(length, filesize)

        self.assertTrue((arr_0 == arr_1).all())

        sfcs_utils.sfcs_delete_file(filepath)

    def test_pipeline(self):
        self._run_pipeline()


class TestSFCSLoad(TestCase):
    @classmethod
    def setUpClass(cls):
        init_sfcs_env()

        # key / iv
        os.environ['VETURBOIO_KEY'] = base64.b64encode(b'abcdefgh12345678').decode('ascii')
        os.environ['VETURBOIO_IV'] = base64.b64encode(b'1234567887654321').decode('ascii')
        # kms info
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

        cls.filepath_0 = "sfcs://model.safetensors"
        cls.filepath_1 = "sfcs://model.pt"
        # mock /tmp as efs mount path
        cls.filepath_2 = "/model.safetensors"
        cls.tensors_0 = {
            "weight1": torch.ones(500, 50),
            "weight2": torch.zeros(500, 50),
        }

        class MockModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.linear1 = torch.nn.Linear(500, 50)
                self.linear2 = torch.nn.Linear(500, 50)

        cls.model = MockModel()

        if torch.cuda.is_available():
            cls.cuda_tensors_0 = deepcopy(cls.tensors_0)
            for key in cls.cuda_tensors_0.keys():
                cls.cuda_tensors_0[key] = cls.cuda_tensors_0[key].cuda()

            cls.cuda_model = MockModel().cuda()

    @classmethod
    def tearDownClass(cls):
        sfcs_utils.sfcs_delete_file(cls.filepath_0[6:])
        sfcs_utils.sfcs_delete_file(cls.filepath_1[6:])

    def _run_pipeline(self, tensors, model, map_location, use_cipher):
        veturboio.save_file(tensors, self.filepath_0, use_cipher=use_cipher)
        loaded_tensors = veturboio.load(self.filepath_0, map_location=map_location, use_cipher=use_cipher)
        for key in tensors.keys():
            self.assertTrue(torch.allclose(tensors[key], loaded_tensors[key]))

        veturboio.save_model(model, self.filepath_0, use_cipher=use_cipher)
        loaded_tensors = veturboio.load(self.filepath_0, map_location=map_location, use_cipher=use_cipher)
        state_dict = model.state_dict()
        for key in state_dict.keys():
            self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))

        veturboio.save_pt(state_dict, self.filepath_1, use_cipher=use_cipher)
        loaded_tensors = veturboio.load(self.filepath_1, map_location=map_location, use_cipher=use_cipher)
        for key in state_dict.keys():
            self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))

        os.environ['VETURBOIO_USE_SFCS_SDK'] = '1'
        loaded_tensors = veturboio.load(self.filepath_2, map_location=map_location, use_cipher=use_cipher)
        del os.environ['VETURBOIO_USE_SFCS_SDK']
        state_dict = model.state_dict()
        for key in state_dict.keys():
            self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))

    def test_pipeline_cpu(self):
        self._run_pipeline(self.tensors_0, self.model, "cpu", use_cipher=False)
        self._run_pipeline(self.tensors_0, self.model, "cpu", use_cipher=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pipeline_cuda(self):
        self._run_pipeline(self.cuda_tensors_0, self.cuda_model, "cuda:0", use_cipher=False)
        self._run_pipeline(self.cuda_tensors_0, self.cuda_model, "cuda:0", use_cipher=True)

    def test_pipeline_cipher_header_cpu(self):
        os.environ["VETURBOIO_CIPHER_HEADER"] = "1"
        self._run_pipeline(self.tensors_0, self.model, "cpu", use_cipher=True)
        del os.environ["VETURBOIO_CIPHER_HEADER"]

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pipeline_cipher_header_cuda(self):
        os.environ["VETURBOIO_CIPHER_HEADER"] = "1"
        self._run_pipeline(self.cuda_tensors_0, self.cuda_model, "cuda:0", use_cipher=True)
        del os.environ["VETURBOIO_CIPHER_HEADER"]
