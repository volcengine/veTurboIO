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


class TestConvertUtil(TestCase):
    def test_convert(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.pt")
            torch.save(self.tensors, filepath)
            convertpath = os.path.join(tmpdirname, "model.safetensors")

            print(f"python -m veturboio.convert -i {filepath} -o {convertpath}")
            os.system(f"python -m veturboio.convert -i {filepath} -o {convertpath}")

            loaded_tensors = veturboio.load(convertpath)
            for key in self.tensors.keys():
                self.assertTrue(torch.allclose(self.tensors[key], loaded_tensors[key]))

    @classmethod
    def setUpClass(cls):
        cls.tensors = {
            "weight1": torch.randn(20, 10),
            "weight2": torch.randn(20, 10),
        }
