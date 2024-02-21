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
import tempfile
import unittest
from copy import deepcopy
from unittest import TestCase

import torch

import veturboio


class TestLoad(TestCase):
    @classmethod
    def setUpClass(cls):
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

        if torch.cuda.is_available():
            cls.cuda_tensors_0 = deepcopy(cls.tensors_0)
            cls.cuda_tensors_1 = deepcopy(cls.tensors_1)

            for key in cls.cuda_tensors_0.keys():
                cls.cuda_tensors_0[key] = cls.cuda_tensors_0[key].cuda()
            for key in cls.cuda_tensors_1.keys():
                cls.cuda_tensors_1[key] = cls.cuda_tensors_1[key].cuda()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _run_pipeline(self, tensors, filepath, map_location):
        loaded_tensors = veturboio.load(filepath, map_location=map_location)
        for key in tensors.keys():
            self.assertTrue(torch.allclose(tensors[key], loaded_tensors[key]))
        return loaded_tensors

    def test_pipeline_cpu(self):
        self._run_pipeline(self.tensors_0, self.filepath_0, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pipeline_cuda(self):
        self._run_pipeline(self.cuda_tensors_0, self.filepath_0, "cuda:0")

    def test_read_multi_state_dict_cpu(self):
        load_tensor_0 = self._run_pipeline(self.tensors_0, self.filepath_0, "cpu")
        load_tensor_1 = self._run_pipeline(self.tensors_1, self.filepath_1, "cpu")

        self.assertEqual(len(load_tensor_0), 2)
        self.assertEqual(len(load_tensor_1), 3)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_read_multi_state_dict_cuda(self):
        load_tensor_0 = self._run_pipeline(self.cuda_tensors_0, self.filepath_0, "cuda:0")
        load_tensor_1 = self._run_pipeline(self.cuda_tensors_1, self.filepath_1, "cuda:0")

        self.assertEqual(len(load_tensor_0), 2)
        self.assertEqual(len(load_tensor_1), 3)

    def test_load_pt_cpu(self):
        loaded_tensors = veturboio.load(self.pt_filepath, map_location="cpu")
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors[key]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_load_pt_cuda(self):
        loaded_tensors = veturboio.load(self.pt_filepath, map_location="cuda:0")

        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.cuda_tensors_0[key], loaded_tensors[key]))
