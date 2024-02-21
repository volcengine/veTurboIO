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
from unittest import TestCase

import torch

import veturboio


class TestAssertException(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_modify_use_pinmem_attr(self):
        helper = veturboio.init_io_helper()
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.safetensors")
            veturboio.save_file(self.tensors, filepath)

            with self.assertRaises(Exception) as context:
                veturboio.load(filepath, map_location="cuda:0", use_pinmem=False, helper=helper)
                veturboio.load(filepath, map_location="cuda:0", use_pinmem=True, helper=helper)
            self.assertTrue(
                'use_pinmem attribute of an exising IOHelper should not be changed' in str(context.exception)
            )

    @classmethod
    def setUpClass(cls):
        cls.tensors = {
            "weight1": torch.randn(20, 10),
            "weight2": torch.randn(20, 10),
        }
