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

import torch

import veturboio


class TestLoad(TestCase):
    @classmethod
    def setUpClass(cls):
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

        cls.tempdir = tempfile.TemporaryDirectory()

        cls.tensors_0 = {
            "weight1": torch.randn(2000, 10),
            "weight2": torch.randn(2000, 10),
        }

        cls.tensors_1 = {
            "weight1": torch.randn(2000, 10),
            "weight2": torch.randn(2000, 10),
            "weight3": torch.randn(2000, 10),
        }

        cls.filepath_0 = os.path.join(cls.tempdir.name, "model_0.safetensors")
        cls.filepath_1 = os.path.join(cls.tempdir.name, "model_1.safetensors")
        veturboio.save_file(cls.tensors_0, cls.filepath_0)
        veturboio.save_file(cls.tensors_1, cls.filepath_1)

        cls.pt_filepath = os.path.join(cls.tempdir.name, "model.pt")
        torch.save(cls.tensors_0, cls.pt_filepath)

        # cipher
        os.environ["VETURBOIO_KEY"] = base64.b64encode(b"abcdefgh12345678").decode("ascii")
        os.environ["VETURBOIO_IV"] = base64.b64encode(b"1234567887654321").decode("ascii")

        cls.filepath_0_enc = os.path.join(cls.tempdir.name, "model_0_enc.safetensors")
        cls.filepath_1_enc = os.path.join(cls.tempdir.name, "model_1_enc.safetensors")
        veturboio.save_file(cls.tensors_0, cls.filepath_0_enc, use_cipher=True)
        veturboio.save_file(cls.tensors_1, cls.filepath_1_enc, use_cipher=True)

        cls.pt_filepath_enc = os.path.join(cls.tempdir.name, "model_enc.pt")
        veturboio.save_pt(cls.tensors_0, cls.pt_filepath_enc, use_cipher=True)

        # cipher with header
        os.environ["VETURBOIO_CIPHER_HEADER"] = "1"
        cls.filepath_0_enc_h = os.path.join(cls.tempdir.name, "model_0_enc_h.safetensors")
        veturboio.save_file(cls.tensors_0, cls.filepath_0_enc_h, use_cipher=True)

        cls.pt_filepath_enc_h = os.path.join(cls.tempdir.name, "model_enc_h.pt")
        veturboio.save_pt(cls.tensors_0, cls.pt_filepath_enc_h, use_cipher=True)

        if torch.cuda.is_available():
            cls.cuda_tensors_0 = deepcopy(cls.tensors_0)
            cls.cuda_tensors_1 = deepcopy(cls.tensors_1)

            for key in cls.cuda_tensors_0.keys():
                cls.cuda_tensors_0[key] = cls.cuda_tensors_0[key].cuda()
            for key in cls.cuda_tensors_1.keys():
                cls.cuda_tensors_1[key] = cls.cuda_tensors_1[key].cuda()

    @classmethod
    def tearDownClass(cls):
        # cls.tempdir.cleanup()
        pass

    def _run_pipeline(self, tensors, filepath, map_location, use_cipher, enable_fast_mode=True):
        loaded_tensors = veturboio.load(
            filepath, map_location=map_location, use_cipher=use_cipher, enable_fast_mode=enable_fast_mode
        )
        for key in tensors.keys():
            self.assertTrue(torch.allclose(tensors[key], loaded_tensors[key]))
        return loaded_tensors

    def test_pipeline_cpu(self):
        self._run_pipeline(self.tensors_0, self.filepath_0, "cpu", use_cipher=False)
        self._run_pipeline(self.tensors_0, self.filepath_0_enc, "cpu", use_cipher=True)
        self._run_pipeline(self.tensors_0, self.filepath_0, "cpu", use_cipher=False, enable_fast_mode=False)
        self._run_pipeline(self.tensors_0, self.filepath_0_enc, "cpu", use_cipher=True, enable_fast_mode=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pipeline_cuda(self):
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0, "cuda:0", use_cipher=False)
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0_enc, "cuda:0", use_cipher=True)
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0, "cuda:0", use_cipher=False, enable_fast_mode=False)
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0_enc, "cuda:0", use_cipher=True, enable_fast_mode=False)

    def test_read_multi_state_dict_cpu(self):
        load_tensor_0 = self._run_pipeline(self.tensors_0, self.filepath_0, "cpu", use_cipher=False)
        load_tensor_1 = self._run_pipeline(self.tensors_1, self.filepath_1, "cpu", use_cipher=False)

        self.assertEqual(len(load_tensor_0), 2)
        self.assertEqual(len(load_tensor_1), 3)

        load_tensor_0_enc = self._run_pipeline(self.tensors_0, self.filepath_0_enc, "cpu", use_cipher=True)
        load_tensor_1_enc = self._run_pipeline(self.tensors_1, self.filepath_1_enc, "cpu", use_cipher=True)

        self.assertEqual(len(load_tensor_0_enc), 2)
        self.assertEqual(len(load_tensor_1_enc), 3)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_read_multi_state_dict_cuda(self):
        load_tensor_0 = self._run_pipeline(self.cuda_tensors_0, self.filepath_0, "cuda:0", use_cipher=False)
        load_tensor_1 = self._run_pipeline(self.cuda_tensors_1, self.filepath_1, "cuda:0", use_cipher=False)

        self.assertEqual(len(load_tensor_0), 2)
        self.assertEqual(len(load_tensor_1), 3)

        load_tensor_0_enc = self._run_pipeline(self.cuda_tensors_0, self.filepath_0_enc, "cuda:0", use_cipher=True)
        load_tensor_1_enc = self._run_pipeline(self.cuda_tensors_1, self.filepath_1_enc, "cuda:0", use_cipher=True)

        self.assertEqual(len(load_tensor_0_enc), 2)
        self.assertEqual(len(load_tensor_1_enc), 3)

    def test_load_pt_cpu(self):
        loaded_tensors = veturboio.load(self.pt_filepath, map_location="cpu", use_cipher=False)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors[key]))

        loaded_tensors_enc = veturboio.load(self.pt_filepath_enc, map_location="cpu", use_cipher=True)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors_enc[key]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_load_pt_cuda(self):
        loaded_tensors = veturboio.load(self.pt_filepath, map_location="cuda:0", use_cipher=False)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.cuda_tensors_0[key], loaded_tensors[key]))

        loaded_tensors_enc = veturboio.load(self.pt_filepath_enc, map_location="cuda:0", use_cipher=True)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.cuda_tensors_0[key], loaded_tensors_enc[key]))

    def test_load_cipher_header_cpu(self):
        os.environ["VETURBOIO_CIPHER_HEADER"] = "1"
        self._run_pipeline(self.tensors_0, self.filepath_0_enc_h, "cpu", use_cipher=True)
        self._run_pipeline(self.tensors_0, self.pt_filepath_enc_h, "cpu", use_cipher=True)
        self._run_pipeline(self.tensors_0, self.filepath_0_enc_h, "cpu", use_cipher=True, enable_fast_mode=False)
        self._run_pipeline(self.tensors_0, self.pt_filepath_enc_h, "cpu", use_cipher=True, enable_fast_mode=False)
        del os.environ["VETURBOIO_CIPHER_HEADER"]

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_load_cipher_header_cuda(self):
        os.environ["VETURBOIO_CIPHER_HEADER"] = "1"
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0_enc_h, "cuda:0", use_cipher=True)
        self._run_pipeline(self.cuda_tensors_0, self.pt_filepath_enc_h, "cuda:0", use_cipher=True)
        self._run_pipeline(
            self.cuda_tensors_0, self.filepath_0_enc_h, "cuda:0", use_cipher=True, enable_fast_mode=False
        )
        self._run_pipeline(
            self.cuda_tensors_0, self.pt_filepath_enc_h, "cuda:0", use_cipher=True, enable_fast_mode=False
        )
        del os.environ["VETURBOIO_CIPHER_HEADER"]

    def test_load_directIO_fall_back(self):
        with tempfile.NamedTemporaryFile(dir="/dev/shm") as tmpFile:
            veturboio.save_file(self.tensors_0, tmpFile.file.name)
            tmpFile.flush()
            loaded_tensors = veturboio.load(tmpFile.name, map_location="cpu", use_direct_io=True)
            for key in self.tensors_0.keys():
                self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors[key]))
