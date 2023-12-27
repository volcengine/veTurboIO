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
import json
import os
import tempfile
import unittest
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

import veturboio
import veturboio.ops.sfcs_utils as sfcs_utils
from veturboio.ops.consts import (
    DEFAULT_CREDENTIAL_PATH_ENV,
    MLP_ACCESS_KEY_FILENAME,
    MLP_SECRET_KEY_FILENAME,
    RDMA_NIC_ENV,
    RDMA_SEGMENT_ENV,
    SFCS_DEFAULT_CONFIG_PATH_ENV,
)
from veturboio.types import FILE_PATH


def init_sfcs_env(ak=None, sk=None):
    os.environ['SFCS_FSNAME'] = 'byted-cpu-sfcs'
    os.environ['SFCS_REGION'] = 'cn-beijing'
    os.environ['SFCS_ACCESS_KEY'] = os.getenv('CI_SFCS_AK', ak)
    os.environ['SFCS_SECRET_KEY'] = os.getenv('CI_SFCS_SK', sk)
    os.environ['SFCS_AUTHENTICATION_SERVICE_NAME'] = 'cfs'
    os.environ['SFCS_NS_ID'] = '18014398509481988'
    os.environ['SFCS_UFS_PATH'] = 'tos://yinzq-bucket/'
    os.environ['SFCS_MULTI_NIC_WHITELIST'] = 'eth0'
    os.environ['SFCS_NETWORK_SEGMENT'] = '172.31.128.0/17'
    os.environ['SFCS_NAMENODE_ENDPOINT_ADDRESS'] = '100.67.19.231'
    os.environ['SFCS_LOG_SEVERITY'] = 'ERROR'
    os.environ['SFCS_CONCAT_DIR'] = '/temp'


def unset_sfcs_env():
    env_vars = [
        'SFCS_FSNAME',
        'SFCS_REGION',
        'SFCS_ACCESS_KEY',
        'SFCS_SECRET_KEY',
        'SFCS_AUTHENTICATION_SERVICE_NAME',
        'SFCS_NS_ID',
        'SFCS_UFS_PATH',
        'SFCS_MULTI_NIC_WHITELIST',
        'SFCS_NETWORK_SEGMENT',
        'SFCS_NAMENODE_ENDPOINT_ADDRESS',
        'SFCS_LOG_SEVERITY',
        'SFCS_CONCAT_DIR',
        'SFCS_SYNC_INTERVAL',
        'SFCS_ENABLE_DNS',
        'SFCS_TASK_ID',
        'SFCS_BLOCK_SIZE',
    ]
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]


class TestSfcsConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        unset_sfcs_env()

    def test_from_env(self):
        env_dict = {
            'SFCS_FSNAME': 'my_fs',
            'SFCS_REGION': 'us-west-1',
            'SFCS_UFS_PATH': '/mnt/data',
            'SFCS_NS_ID': 'ns-12345',
            'SFCS_ACCESS_KEY': 'my_access_key',
            'SFCS_SECRET_KEY': 'my_secret_key',
            'SFCS_NETWORK_SEGMENT': 'segment1',
            'SFCS_MULTI_NIC_WHITELIST': 'nic1,nic2',
            'SFCS_LOG_SEVERITY': 'INFO',
            'SFCS_TASK_ID': 'task-12345',
            'SFCS_SYNC_INTERVAL': '50',
            'SFCS_NAMENODE_ENDPOINT_ADDRESS': 'http://namenode:8020',
            'SFCS_AUTHENTICATION_SERVICE_NAME': 'cfs-service',
            'SFCS_ENABLE_DNS': 'True',
            'SFCS_BLOCK_SIZE': '123',
        }
        for key, val in env_dict.items():
            os.environ[key] = val
        config = sfcs_utils.SfcsConfiguration.from_env("test")
        self.assertEqual(config.fsname, 'my_fs')
        self.assertEqual(config.region, 'us-west-1')
        self.assertEqual(config.ufs_path, '/mnt/data')
        self.assertEqual(config.ns_id, 'ns-12345')
        self.assertEqual(config.access_key, 'my_access_key')
        self.assertEqual(config.secret_key, 'my_secret_key')
        self.assertEqual(config.network_segment, 'segment1')
        self.assertEqual(config.multi_nic_whitelist, 'nic1,nic2')
        self.assertEqual(config.log_severity, 'INFO')
        self.assertEqual(config.task_id, 'task-12345')
        self.assertEqual(config.sync_interval, '50')
        self.assertEqual(config.namenode_endpoint_address, 'http://namenode:8020')
        self.assertEqual(config.authentication_service_name, 'cfs-service')
        self.assertEqual(config.block_size, '123')
        self.assertTrue(config.enable_dns)
        unset_sfcs_env()

    def test_minimal_config(self):
        env_dict = {
            'SFCS_FSNAME': 'my_fs',
            'SFCS_REGION': 'us-west-1',
            'SFCS_UFS_PATH': '/mnt/data',
            'SFCS_NS_ID': 'ns-12345',
            'SFCS_NETWORK_SEGMENT': 'segment1',
            'SFCS_MULTI_NIC_WHITELIST': 'nic1,nic2',
            'SFCS_AUTHENTICATION_SERVICE_NAME': 'cfs-service',
        }
        for key, val in env_dict.items():
            os.environ[key] = val
        config = sfcs_utils.SfcsConfiguration.from_env("test")
        self.assertEqual(config.fsname, 'my_fs')
        self.assertEqual(config.region, 'us-west-1')
        self.assertEqual(config.ufs_path, '/mnt/data')
        self.assertEqual(config.ns_id, 'ns-12345')
        self.assertEqual(config.access_key, '')
        self.assertEqual(config.secret_key, '')
        self.assertEqual(config.network_segment, 'segment1')
        self.assertEqual(config.multi_nic_whitelist, 'nic1,nic2')
        self.assertEqual(config.log_severity, 'INFO')
        self.assertEqual(config.task_id, 'sfcs')
        self.assertEqual(config.sync_interval, '-1')
        self.assertEqual(config.namenode_endpoint_address, '')
        self.assertEqual(config.authentication_service_name, 'cfs-service')
        self.assertEqual(config.enable_dns, 'false')
        self.assertEqual(config.block_size, '')

    def test_json_from_env(self):
        env_json_dict = {
            'SFCS_FSNAME': json.dumps({'test': 'my_fs', 'other_key': 'other_fs'}),
            'SFCS_UFS_PATH': json.dumps({'test': '/mnt/data', 'other_key': '/mnt/other_data'}),
            'SFCS_NS_ID': json.dumps({'test': 'ns-12345', 'other_key': 'ns-67890'}),
        }
        env_dict = {
            'SFCS_REGION': 'us-west-1',
            'SFCS_ACCESS_KEY': 'my_access_key',
            'SFCS_SECRET_KEY': 'my_secret_key',
            'SFCS_NETWORK_SEGMENT': 'segment1',
            'SFCS_MULTI_NIC_WHITELIST': 'nic1,nic2',
            'SFCS_LOG_SEVERITY': 'INFO',
            'SFCS_TASK_ID': 'task-12345',
            'SFCS_SYNC_INTERVAL': '50',
            'SFCS_NAMENODE_ENDPOINT_ADDRESS': 'http://namenode:8020',
            'SFCS_AUTHENTICATION_SERVICE_NAME': 'cfs-service',
            'SFCS_ENABLE_DNS': 'True',
        }
        for key, val in env_json_dict.items():
            os.environ[key] = val
        for key, val in env_dict.items():
            os.environ[key] = val

        config = sfcs_utils.SfcsConfiguration.from_env("test")

        self.assertEqual(config.fsname, 'my_fs')
        self.assertEqual(config.region, 'us-west-1')
        self.assertEqual(config.ufs_path, '/mnt/data')
        self.assertEqual(config.ns_id, 'ns-12345')
        self.assertEqual(config.access_key, 'my_access_key')
        self.assertEqual(config.secret_key, 'my_secret_key')
        self.assertEqual(config.network_segment, 'segment1')
        self.assertEqual(config.multi_nic_whitelist, 'nic1,nic2')
        self.assertEqual(config.log_severity, 'INFO')
        self.assertEqual(config.task_id, 'task-12345')
        self.assertEqual(config.sync_interval, '50')
        self.assertEqual(config.namenode_endpoint_address, 'http://namenode:8020')
        self.assertEqual(config.authentication_service_name, 'cfs-service')
        self.assertTrue(config.enable_dns)
        unset_sfcs_env()


class TestSfcsConfigurationOverride(unittest.TestCase):
    def test_override(self):
        config1 = sfcs_utils.SfcsConfiguration(
            fsname='my_fs_1',
            region='us-west-1',
            ufs_path='/mnt/data1',
            ns_id='ns-12345',
            access_key='key1',
            secret_key='secret1',
            network_segment='segment1',
            multi_nic_whitelist='nic1',
            log_severity='DEBUG',
            task_id='task-1',
            namenode_endpoint_address='http://namenode1:8020',
            authentication_service_name='service1',
            enable_dns=True,
            block_size='222',
        )

        config2 = sfcs_utils.SfcsConfiguration(
            fsname='my_fs_2',
            region='us-west-2',
            ufs_path='/mnt/data2',
            ns_id='ns-67890',
            access_key='key2',
            secret_key='secret2',
            network_segment='segment2',
            multi_nic_whitelist='nic2',
            log_severity='INFO',
            task_id='task-2',
            sync_interval='30',
            namenode_endpoint_address='http://namenode2:8020',
            authentication_service_name='service2',
            enable_dns=False,
            block_size='111',
        )

        result = sfcs_utils.SfcsConfiguration.override(config1, config2)

        # config1 的值应当覆盖 config2 的值
        self.assertEqual(result.fsname, 'my_fs_1')
        self.assertEqual(result.region, 'us-west-1')
        self.assertEqual(result.ufs_path, '/mnt/data1')
        self.assertEqual(result.ns_id, 'ns-12345')
        self.assertEqual(result.access_key, 'key1')
        self.assertEqual(result.secret_key, 'secret1')
        self.assertEqual(result.network_segment, 'segment1')
        self.assertEqual(result.multi_nic_whitelist, 'nic1')
        self.assertEqual(result.log_severity, 'DEBUG')
        self.assertEqual(result.task_id, 'task-1')
        self.assertEqual(result.sync_interval, '30')
        self.assertEqual(result.namenode_endpoint_address, 'http://namenode1:8020')
        self.assertEqual(result.authentication_service_name, 'service1')
        self.assertTrue(result.enable_dns)
        self.assertEqual(result.block_size, '222')


class TestXMLGeneration(TestCase):
    def setUp(self):
        unset_sfcs_env()

    def tearDown(self):
        unset_sfcs_env()

    def test_block_generation(self):
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        init_sfcs_env("akkk", "skkk")
        os.environ['SFCS_BLOCK_SIZE'] = '2112'
        old_func = sfcs_utils.generate_sfcs_conf_xml
        sfcs_property = {}
        sfcs_utils.generate_sfcs_conf_xml = hack_generate_property(sfcs_property)
        group = sfcs_utils.init_sfcs_conf('/tmp/test.safetensor')
        self.assertEqual(group, 'tmp')
        self.assertEqual(sfcs_property['cfs.filesystem.ufs-path'], 'tos://yinzq-bucket/')
        self.assertEqual(sfcs_property['cfs.access.key'], os.getenv('CI_SFCS_AK', "akkk"))
        self.assertEqual(sfcs_property['cfs.secret.key'], os.getenv('CI_SFCS_SK', "skkk"))
        self.assertEqual(sfcs_property['dfs.default.blocksize'], '2112')
        sfcs_utils.generate_sfcs_conf_xml = old_func

    def test_load_default(self):
        data = get_json_data()
        credential_path, _, _ = create_ak_sk_files()
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
            json.dump(data, temp_file, indent=4)
            temp_file_path = temp_file.name
        os.environ[DEFAULT_CREDENTIAL_PATH_ENV] = credential_path.name
        os.environ[SFCS_DEFAULT_CONFIG_PATH_ENV] = temp_file_path
        init_rdma_nic_env()
        default_configuration = sfcs_utils.SfcsDefaultConfig.init()
        expected_configuration = get_expect_default_config()
        self.assertDictEqual(expected_configuration.cache_policies, default_configuration.cache_policies)
        self.assertEqual(expected_configuration.nic_white_list, default_configuration.nic_white_list)
        self.assertEqual(expected_configuration.network_segment, default_configuration.network_segment)
        self.assertEqual(expected_configuration.access_key, default_configuration.access_key)
        self.assertEqual(expected_configuration.secret_key, default_configuration.secret_key)
        self.assertEqual(len(expected_configuration.configurations), 2)
        os.remove(temp_file_path)
        credential_path.cleanup()

    def test_init_config(self):
        expected_xml_name_for_testmodel = sfcs_utils.generate_sfcs_conf_path("/sfcs/testmodel")
        expected_xml_name_for_required = sfcs_utils.generate_sfcs_conf_path("/sfcs/required")
        self.assertTrue("sfcs#testmodel" in expected_xml_name_for_testmodel)
        self.assertTrue("sfcs#required" in expected_xml_name_for_required)
        try:
            os.remove(expected_xml_name_for_testmodel)
            os.remove(expected_xml_name_for_required)
        except FileNotFoundError as e:
            pass

        sfcs_utils.sfcs_default_config = get_expect_default_config()

        group = sfcs_utils.init_sfcs_conf('/sfcs/required/ckpt/xx.safetensor')
        self.assertEqual(group, '/sfcs/required')
        self.assertEqual(os.path.exists(expected_xml_name_for_required), True)
        group = sfcs_utils.init_sfcs_conf('/sfcs/testmodel/ckpt/xx.safetensor')
        self.assertEqual(group, '/sfcs/testmodel')

        self.assertEqual(os.path.exists(expected_xml_name_for_testmodel), True)

        sfcs_utils.sfcs_default_config = get_expect_default_config()
        with self.assertRaises(sfcs_utils.CredentialError):
            sfcs_utils.sfcs_default_config.configurations["/sfcs/testmodel"].access_key = ''
            sfcs_utils.init_sfcs_conf('/sfcs/testmodel/ckpt/xx.safetensor/')

        sfcs_utils.sfcs_default_config = get_expect_default_config()
        with self.assertRaises(sfcs_utils.NetworkError):
            sfcs_utils.sfcs_default_config.configurations["/sfcs/testmodel"].enable_dns = "false"
            sfcs_utils.init_sfcs_conf('/sfcs/testmodel/ckpt/xx.safetensor/')

        os.remove(expected_xml_name_for_required)
        os.remove(expected_xml_name_for_testmodel)

    def test_init_config_with_invalid_path(self):
        # should fall back to original method
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        with self.assertRaises(ValueError):
            sfcs_utils.init_sfcs_conf('.safetensor')

        with self.assertRaises(ValueError):
            sfcs_utils.init_sfcs_conf('/sfcs/invalid/ckpt/xx.safetensor/')

    def test_override_config_for_no_policy_hit(self):
        xmlFile = sfcs_utils.generate_sfcs_conf_path('tmp')
        try:
            os.remove(xmlFile)
        except FileNotFoundError as e:
            pass
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        init_sfcs_env("akkk", "skkk")
        old_func = sfcs_utils.generate_sfcs_conf_xml
        sfcs_property = {}
        sfcs_utils.generate_sfcs_conf_xml = hack_generate_property(sfcs_property)
        group = sfcs_utils.init_sfcs_conf('/tmp/test.safetensor')
        self.assertEqual(group, 'tmp')
        self.assertEqual(sfcs_property['cfs.filesystem.ufs-path'], 'tos://yinzq-bucket/')
        self.assertEqual(sfcs_property['cfs.access.key'], os.getenv('CI_SFCS_AK', "akkk"))
        self.assertEqual(sfcs_property['cfs.secret.key'], os.getenv('CI_SFCS_SK', "skkk"))
        sfcs_utils.generate_sfcs_conf_xml = old_func

    def test_override_config_for_policy_hit(self):
        xmlFile = sfcs_utils.generate_sfcs_conf_path('/sfcs/testmodel/ckpt/xx.safetensor')
        try:
            os.remove(xmlFile)
        except FileNotFoundError as e:
            pass
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        old_func = sfcs_utils.generate_sfcs_conf_xml
        sfcs_property = {}
        init_sfcs_env("akkk", "skkk")
        sfcs_utils.generate_sfcs_conf_xml = hack_generate_property(sfcs_property)
        group = sfcs_utils.init_sfcs_conf('/sfcs/testmodel/ckpt/xx.safetensor')
        self.assertEqual(group, '/sfcs/testmodel')
        self.assertEqual(sfcs_property['cfs.filesystem.ufs-path'], 'tos://yinzq-bucket/')
        self.assertEqual(sfcs_property['cfs.access.key'], os.getenv('CI_SFCS_AK', "akkk"))
        self.assertEqual(sfcs_property['cfs.secret.key'], os.getenv('CI_SFCS_SK', "skkk"))
        sfcs_utils.generate_sfcs_conf_xml = old_func


class TestPathMapper(TestCase):
    def test_for_policy_hit(self):
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        valid_path = sfcs_utils.path_mapper("/sfcs/testmodel/ckpt/xx.safetensor", "/sfcs/testmodel")
        self.assertEqual(valid_path, "/ckpt/xx.safetensor")
        valid_path = sfcs_utils.path_mapper("/sfcs/required/ckpt/xx.safetensor", "/sfcs/required")
        self.assertEqual(valid_path, "/sft_demo/ckpt/xx.safetensor")

    def test_for_policy_not_hit(self):
        sfcs_utils.sfcs_default_config = get_expect_default_config()
        original_path = sfcs_utils.path_mapper("/xx/ckpt/xx.safetensor", "/xx")
        self.assertEqual(original_path, "/xx/ckpt/xx.safetensor")
        original_path = sfcs_utils.path_mapper("/xx/ckpt/xx.safetensor", "xx")
        self.assertEqual(original_path, "/xx/ckpt/xx.safetensor")


class TestSFCS(TestCase):
    @classmethod
    def setUpClass(cls):
        init_sfcs_env()

    def _run_pipeline(self):
        filepath = "/data.bin"
        filesize = 1024 * 1024

        # first_path = os.path.abspath(filepath).split("/")[1]
        # sfcs_conf = os.path.join(os.getcwd(), first_path + '.xml')
        group = sfcs_utils.sfcs_conf_group(filepath)
        sfcs_conf = sfcs_utils.generate_sfcs_conf_path(group)
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

        filepaths = ["/data1.bin", "/data2.bin"]
        filesizes = [1024 * 1024, 1024 * 2048]
        tensors_w = [torch.ones(filesizes[i], dtype=torch.uint8) for i in range(2)]
        tensors_r = [torch.empty(filesizes[i], dtype=torch.uint8) for i in range(2)]

        for filepath in filepaths:
            sfcs_utils.sfcs_delete_file(filepath)

        sfcs_utils.sfcs_write_multi_files(filepaths, tensors_w)

        lengths = sfcs_utils.sfcs_get_multi_file_size(filepaths)
        self.assertTrue(filesizes == lengths)

        sfcs_utils.sfcs_read_multi_files(filepaths, tensors_r)

        for filepath in filepaths:
            sfcs_utils.sfcs_delete_file(filepath)

        for i in range(2):
            self.assertTrue(torch.allclose(tensors_w[i], tensors_r[i]))

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

        shmem = veturboio.load_to_shmem(self.filepath_0, use_cipher=use_cipher)
        loaded_tensors = veturboio.load(
            os.path.join("/dev/shm/", shmem.name), map_location=map_location, enable_fast_mode=False, use_cipher=False
        )
        for key in tensors.keys():
            self.assertTrue(torch.allclose(tensors[key], loaded_tensors[key]))
        shmem.close()
        shmem.unlink()

        # pre allocated tensors
        pre_allocated_tensors = {
            "weight1": torch.zeros(500, 50),
            "weight2": torch.ones(500, 50),
        }
        if not map_location == "cpu":
            for key in pre_allocated_tensors.keys():
                pre_allocated_tensors[key] = pre_allocated_tensors[key].cuda()
        veturboio.save_file(tensors, self.filepath_0, use_cipher=use_cipher, enable_fast_mode=True)
        loaded_tensors = veturboio.load(
            self.filepath_0, map_location=map_location, use_cipher=use_cipher, state_dict=pre_allocated_tensors
        )
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

    def test_sfcs_write(self):
        tensors = {
            "empty": torch.Tensor([]),
            "nested_empty": torch.Tensor([[]]),
            "non_empty": torch.Tensor([i for i in range(0, 64)]),
            "empty_bool": torch.BoolTensor([]),
            "non_empty_bool": torch.BoolTensor([True for i in range(0, 64)]),
        }

        # Test common write and concat write
        def save_file_wrapper(path: str, fast_mode: bool, use_cipher: bool):
            veturboio.save_file(tensors, path, enable_fast_mode=fast_mode, use_cipher=use_cipher)
            load_tensors = veturboio.load(path, map_location="cpu", use_cipher=use_cipher, enable_fast_mode=True)
            for key, val in tensors.items():
                self.assertTrue(torch.allclose(val, load_tensors[key]))

        save_file_wrapper(self.filepath_0, True, True)
        save_file_wrapper(self.filepath_0, True, False)
        save_file_wrapper(self.filepath_0, False, True)
        save_file_wrapper(self.filepath_0, False, False)


def create_ak_sk_files():
    temp_dir = tempfile.TemporaryDirectory()
    access_key_content = "thisisak"
    secret_key_content = "thisissk"
    access_key_path = os.path.join(temp_dir.name, MLP_ACCESS_KEY_FILENAME)
    secret_key_path = os.path.join(temp_dir.name, MLP_SECRET_KEY_FILENAME)
    with open(access_key_path, 'w') as access_key_file:
        access_key_file.write(access_key_content)
    with open(secret_key_path, 'w') as secret_key_file:
        secret_key_file.write(secret_key_content)
    return temp_dir, access_key_path, secret_key_path


def init_rdma_nic_env():
    os.environ[RDMA_SEGMENT_ENV] = "33.0.0.0/8"
    os.environ[RDMA_NIC_ENV] = "eth2,eth3,eth4"


def hack_generate_property(property_dict):
    def hack(sfcs_conf: FILE_PATH, sfcs_properties: dict):
        for k, v in sfcs_properties.items():
            property_dict[k] = v

    return hack


def get_expect_default_config() -> sfcs_utils.SfcsDefaultConfig:
    data = get_json_data()
    expected_configuration = sfcs_utils.SfcsDefaultConfig()
    cache_policies = {
        key: sfcs_utils.CachePolicy(
            mount_path=value["mount_path"],
            cache_prefix=value["cache_prefix"],
            authentication_service_name=value["authentication_service_name"],
            enable_dns=value["enable_dns"],
            sfcs_fs_name=value["sfcs_fs_name"],
            region=value["region"],
            ufs_path=value["ufs_path"],
            ns_id=value["ns_id"],
        )
        for key, value in data["cache_policies"].items()
    }
    expected_configuration.access_key = "thisisak"
    expected_configuration.secret_key = "thisissk"
    expected_configuration.cache_policies = cache_policies
    expected_configuration.nic_white_list = "eth2,eth3,eth4"
    expected_configuration.network_segment = "33.0.0.0/8"
    expected_configuration.init_sfcs_configuration()
    return expected_configuration


def get_json_data():
    return {
        "kind": "SfcsConfiguration",
        "apiVersion": "v1alpha1",
        "cache_policies": {
            "/sfcs/testmodel": {
                "mount_path": "/sfcs/testmodel",
                "cache_prefix": "/",
                "sfcs_fs_name": "sfcs-2100583650-test",
                "ufs_path": "tos://testmodel/",
                "ns_id": "414331165718085736",
                "region": "cn-shanghai.cfs",
                "authentication_service_name": "cfs",
                "enable_dns": "true",
            },
            "/sfcs/required": {
                "mount_path": "/sfcs/required",
                "cache_prefix": "/sft_demo",
                "sfcs_fs_name": "sfcs-2100583650-test",
                "ufs_path": "tos://2100583650-tos/sft_demo/",
                "ns_id": "414331165718085751",
                "region": "cn-shanghai.cfs",
                "authentication_service_name": "cfs",
                "enable_dns": "true",
            },
        },
    }
