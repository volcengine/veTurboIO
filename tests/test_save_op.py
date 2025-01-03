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
            "weight2": torch.IntTensor(2000, 10),
            "weight3": torch.BoolTensor(2000, 10),
        }

        class MockModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.linear1 = torch.nn.Linear(100, 50)
                self.linear2 = torch.nn.Linear(100, 50)

        cls.model = MockModel()

        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filepath_0 = os.path.join(cls.tempdir.name, "model_0.safetensors")
        cls.filepath_1 = os.path.join(cls.tempdir.name, "model_0.pt")
        cls.filepath_2 = os.path.join(cls.tempdir.name, "model_0_fast.safetensors")
        cls.filepath_3 = os.path.join(cls.tempdir.name, "model_1.safetensors")

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_save_file(self):
        veturboio.save_file(self.tensors_0, self.filepath_0)
        with safe_open(self.filepath_0, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 3
            for key in f.keys():
                self.assertTrue(torch.allclose(self.tensors_0[key], f.get_tensor(key)))

        # enable fast mode
        veturboio.save_file(self.tensors_0, self.filepath_2, enable_fast_mode=True)
        with safe_open(self.filepath_2, framework="pt", device="cpu") as f:
            assert len(f.keys()) == 3
            for key in f.keys():
                self.assertTrue(torch.allclose(self.tensors_0[key], f.get_tensor(key)))

    def test_save_file_for_clone_share_tensors(self):
        share_dict = {"key1": self.tensors_0["weight1"], "key2": self.tensors_0["weight1"]}
        veturboio.save_file(share_dict, self.filepath_0, force_save_shared_tensor=True, force_clone_shared_tensor=True)
        assert len(share_dict) == 2  # assert save_file won't change user's state_dict.
        with safe_open(self.filepath_0, framework="pt", device="cpu") as f:
            for key in f.keys():
                assert key in share_dict
                self.assertTrue(torch.allclose(share_dict[key], f.get_tensor(key)))

    def test_save_model(self):
        veturboio.save_model(self.model, self.filepath_3, use_cipher=True)
        loaded_tensors = veturboio.load(self.filepath_3, map_location="cpu", use_cipher=True)
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))

    def test_save_pt(self):
        veturboio.save_pt(self.tensors_0, self.filepath_1)
        loaded_tensors = torch.load(self.filepath_1)
        for key in self.tensors_0.keys():
            self.assertTrue(torch.allclose(self.tensors_0[key], loaded_tensors[key]))
