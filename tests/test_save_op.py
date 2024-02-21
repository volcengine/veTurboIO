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
from safetensors import safe_open

import veturboio


class TestSave(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensors_0 = {
            "weight1": torch.randn(2000, 10),
            "weight2": torch.randn(2000, 10),
        }

        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filepath_0 = os.path.join(cls.tempdir.name, "model_0.safetensors")
        cls.filepath_1 = os.path.join(cls.tempdir.name, "model_0.pt")

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_save_file(self):
        veturboio.save_file(self.tensors_0, self.filepath_0)
        with safe_open(self.filepath_0, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.assertTrue(torch.allclose(self.tensors_0[key], f.get_tensor(key)))

    def test_save_pt(self):
        veturboio.save_pt(self.tensors_0, self.filepath_1)
        loaded_tensors = torch.load(self.filepath_1)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors[key]))
