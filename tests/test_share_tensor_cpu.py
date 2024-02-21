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
from unittest import TestCase

import torch

import veturboio


class TestSharedTensorLoad(TestCase):
    @classmethod
    def setUpClass(cls):
        class MockModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 10)
                self.linear3 = self.linear2

        cls.model = MockModel()

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.safetensors")
            veturboio.save_model(self.model, filepath)
            loaded_tensors = veturboio.load(filepath, map_location="cpu")

            state_dict = self.model.state_dict()
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))

    def test_save_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.safetensors")
            veturboio.save_file(self.model.state_dict(), filepath, force_save_shared_tensor=True)
            loaded_tensors = veturboio.load(filepath, map_location="cpu")

            state_dict = self.model.state_dict()
            for key in state_dict.keys():
                self.assertTrue(torch.allclose(state_dict[key], loaded_tensors[key]))
