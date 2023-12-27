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
import shutil
import tempfile
import threading
import xml.dom.minidom
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from veturboio.ops.cipher import CipherInfo, DataPipeClient
from veturboio.ops.consts import (
    DEFAULT_CREDENTIAL_PATH,
    DEFAULT_CREDENTIAL_PATH_ENV,
    DEFAULT_NIC_NAME,
    MLP_ACCESS_KEY_FILENAME,
    MLP_SECRET_KEY_FILENAME,
    RDMA_NIC_ENV,
    RDMA_SEGMENT_ENV,
    SFCS_DEFAULT_CONFIG_PATH_ENV,
    SFCS_DEFAULT_METAINFO_PATH,
)
from veturboio.types import FILE_PATH

try:
    from veturboio.utils.load_veturboio_ext import load_veturboio_ext

    veturboio_ext = load_veturboio_ext()
    SFCSFs = veturboio_ext.SFCSFs
    SFCSFile = veturboio_ext.SFCSFile
except ImportError:
    SFCSFs = None
    SFCSFile = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")

SFCS_JSON_ENV_LIST = [
    'SFCS_FSNAME',
    'SFCS_NS_ID',
    'SFCS_UFS_PATH',
]

SFCS_REQ_ENV_LIST = [
    'SFCS_FSNAME',
    'SFCS_REGION',
    'SFCS_AUTHENTICATION_SERVICE_NAME',
    'SFCS_NS_ID',
    'SFCS_UFS_PATH',
    'SFCS_MULTI_NIC_WHITELIST',
    'SFCS_NETWORK_SEGMENT',
]

SFCS_OPT_ENV_LIST = [
    'SFCS_ACCESS_KEY',
    'SFCS_SECRET_KEY',
    'SFCS_NAMENODE_ENDPOINT_ADDRESS',
    'SFCS_LOG_SEVERITY',
    'SFCS_SYNC_INTERVAL',
    'SFCS_ENABLED_DNS',
    'SFCS_TASK_ID',
    'SFCS_BLOCK_SIZE',
]


class CredentialError(ValueError):
    pass


class NetworkError(ValueError):
    pass


def default_sfcs_properties() -> dict:
    return {
        'cfs.filesystem.fs-mode': 'ACC',
        'cfs.filesystem.resolve.addr.by.dns': 'false',
        'cfs.metrics.emitters': 'metric_server;local_prometheus',
        'cfs.client.metadata-cache.enable': 'false',
        'rpc.client.channel.pool.size': '32',
        'dfs.default.replica': '2',
        'cfs.client.multi-nic.enabled': 'true',
        'fs.datanode.router.ignore-main-nic': 'true',
        'cfs.datanode.router.shuffle': 'true',
    }


@dataclass
class CachePolicy:
    mount_path: str = ""
    cache_prefix: str = ""
    authentication_service_name: str = ""
    enable_dns: str = ""
    sfcs_fs_name: str = ""
    region: str = ""
    ufs_path: str = ""
    ns_id: str = ""
    name_node_endpoint: str = ""


@dataclass
class SfcsConfiguration:
    fsname: str = ""
    region: str = ""
    ufs_path: str = ""
    ns_id: str = ""
    access_key: str = ""
    secret_key: str = ""
    network_segment: str = ""
    multi_nic_whitelist: str = ""
    namenode_endpoint_address: str = ""
    authentication_service_name: str = ""
    block_size: str = ""
    task_id: str = "sfcs"
    enable_dns: str = "false"
    log_severity: str = "INFO"
    sync_interval: str = "-1"

    @classmethod
    def from_env(cls, key=None) -> 'SfcsConfiguration':
        init_values = {}
        for field in fields(cls):
            env_var_name = f'SFCS_{field.name.upper()}'
            env_value = os.getenv(env_var_name, str(field.default))
            if env_var_name in SFCS_JSON_ENV_LIST:
                try:
                    json_value = json.loads(env_value)
                    env_value = json_value[key]
                except Exception:
                    logger.warning(
                        f"can't parse env variable as json format or set env by key. variable name: {env_var_name}, key: {key}, value: {env_value}"
                    )
            if field.type == bool:
                init_values[field.name] = str(env_value).lower() in ('true', '1', 't')
            else:
                init_values[field.name] = str(env_value)
        return cls(**init_values)

    @classmethod
    def override(cls, primary: 'SfcsConfiguration', secondary: 'SfcsConfiguration') -> 'SfcsConfiguration':
        override_values = {}
        for field in fields(cls):
            primary_value = getattr(primary, field.name)
            secondary_value = getattr(secondary, field.name)

            if primary_value != field.default:
                override_values[field.name] = primary_value
            else:
                override_values[field.name] = secondary_value

        return cls(**override_values)

    def validate(self, use_datapipe=False):
        validators = [
            getattr(self, method)
            for method in dir(self)
            if (method.startswith('validate_') and not method.endswith("optional"))
        ]
        if not use_datapipe:
            validators.extend([self.validate_credential_optional, self.validate_endpoint_optional])
        for validator in validators:
            validator()

    def validate_network_setting(self):
        if not self.multi_nic_whitelist or not self.network_segment:
            raise NetworkError('Network setting is not valid')
        return

    def validate_endpoint_optional(self):
        if str(self.enable_dns).lower() != 'true' and self.namenode_endpoint_address == "":
            raise NetworkError('name node endpoint should be set setting is not valid')
        return

    def validate_required_fields(self):
        for field in fields(SfcsConfiguration):
            value = getattr(self, field.name)
            if "SFCS_" + field.name.upper() not in SFCS_REQ_ENV_LIST:
                continue
            if value == field.default:
                raise RuntimeError(f'{field.name} is required')

    def validate_credential_optional(self):
        if not self.access_key or not self.secret_key:
            raise CredentialError('Credential is not valid')

    def to_property(self) -> dict:
        properties = default_sfcs_properties()
        properties['dfs.default.uri'] = build_uri(self.fsname, self.region)
        properties['dfs.authentication.service.name'] = self.authentication_service_name
        properties['cfs.filesystem.ns-id'] = self.ns_id
        properties['cfs.filesystem.ufs-path'] = self.ufs_path
        properties['cfs.metrics.server.host'] = build_metric_server_host(self.region)
        properties['cfs.client.multi-nic.whitelist'] = self.multi_nic_whitelist
        properties['cfs.client.network.segment'] = self.network_segment
        properties['dfs.client.log.severity'] = self.log_severity
        properties['cfs.filesystem.resolve.addr.by.dns'] = (
            "false" if str(self.enable_dns).lower() != "true" else "true"
        )
        properties['cfs.filesystem.task-id'] = self.task_id
        properties['cfs.filesystem.sync-interval'] = self.sync_interval
        properties['cfs.access.key'] = self.access_key
        properties['cfs.secret.key'] = self.secret_key
        properties['cfs.namenode.endpoint.address.' + self.fsname] = self.namenode_endpoint_address
        properties['dfs.default.blocksize'] = self.block_size
        trimmed_properties = {k: v for k, v in properties.items() if v is not None and len(v) != 0}
        return trimmed_properties


class SfcsDefaultConfig:
    def __init__(self):
        self.cache_policies: Dict[str, CachePolicy] = {}
        self.network_segment = ""
        self.nic_white_list = ""
        self.access_key = ""
        self.secret_key = ""
        self.configurations: Dict[str, SfcsConfiguration] = {}

    def __repr__(self):
        return self.__dict__.__str__()

    @staticmethod
    def init():
        config = SfcsDefaultConfig()
        config.init_cache_policies_from_file()
        config.init_network_config_from_env()
        config.init_credential_from_file()
        config.init_sfcs_configuration()
        return config

    def init_network_config_from_env(self):
        self.nic_white_list = os.environ.get(RDMA_NIC_ENV, DEFAULT_NIC_NAME)
        if os.environ.get(RDMA_SEGMENT_ENV):
            self.network_segment = os.environ.get(RDMA_SEGMENT_ENV)
        else:

            def get_segment_by_nic_name(nic_name):
                import ipaddress

                import netifaces as ni

                ip_info = ni.ifaddresses(nic_name)[ni.AF_INET][0]
                ip_address = ip_info['addr']
                netmask = ip_info['netmask']
                network = ipaddress.ip_network(f"{ip_address}/{netmask}", strict=False)
                return network

            try:
                self.network_segment = str(get_segment_by_nic_name(self.nic_white_list))
            except Exception as e:
                logger.warning(f"Failed to get network segment from {self.nic_white_list}, error: {e}")

    def init_credential_from_file(self):
        credential_path = os.environ.get(DEFAULT_CREDENTIAL_PATH_ENV, DEFAULT_CREDENTIAL_PATH)
        ak_path = os.path.join(credential_path, MLP_ACCESS_KEY_FILENAME)
        sk_path = os.path.join(credential_path, MLP_SECRET_KEY_FILENAME)
        if os.path.exists(ak_path) and os.path.exists(sk_path):
            with open(ak_path, 'r') as f:
                self.access_key = f.read().strip()
            with open(sk_path, 'r') as f:
                self.secret_key = f.read().strip()

    def init_cache_policies_from_file(self, file_path: str = None):
        file_path = file_path or os.environ.get(SFCS_DEFAULT_CONFIG_PATH_ENV, SFCS_DEFAULT_METAINFO_PATH)
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as file:
            try:
                json_config = json.load(file)
                cache_polices = json_config['cache_policies']
                policy_map = {}
                for key, policy_dict in cache_polices.items():
                    policy = CachePolicy(**policy_dict)
                    policy_map[key] = policy
                self.cache_policies = policy_map
            except Exception as e:
                logger.warning(f'Failed to load SFCSConfiguration from {file_path}, error: {e}')
                return

    def init_sfcs_configuration(
        self,
    ):
        for group, policy in self.cache_policies.items():
            self.configurations[group] = SfcsConfiguration(
                fsname=policy.sfcs_fs_name,
                region=policy.region,
                ufs_path=policy.ufs_path,
                ns_id=policy.ns_id,
                access_key=self.access_key,
                secret_key=self.secret_key,
                network_segment=self.network_segment,
                multi_nic_whitelist=self.nic_white_list,
                enable_dns=policy.enable_dns,
                authentication_service_name=policy.authentication_service_name,
            )

    def get_mount_path_or_root_dir(self, *, path: FILE_PATH) -> str:
        root_path = os.path.abspath(path).split("/")[1]
        if self.cache_policies is None:
            return root_path
        # using the longest mount path
        longest_mount_prefix = ""
        for key in self.cache_policies:
            if path.startswith(key) and len(longest_mount_prefix) < len(key):
                longest_mount_prefix = key
        if len(longest_mount_prefix) == 0:
            # none term hit
            return root_path
        return longest_mount_prefix

    def override_configuration_with_env(self, key) -> SfcsConfiguration:
        env_conf = SfcsConfiguration.from_env(key)
        default_conf = self.configurations.get(key, SfcsConfiguration())
        conf = SfcsConfiguration.override(env_conf, default_conf)
        return conf


sfcs_default_config = SfcsDefaultConfig.init()


class CredentialsHelper:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = {}
        # daemon thread will stop when parent thread exits
        self.threads = {}
        self.stop_flag = threading.Event()
        self.client = None

    def run(self, group: str, sfcs_conf_path: str) -> None:
        if not self.running.get(group, False):
            with self.lock:
                if not self.running.get(group, False):
                    if self.client is None:
                        self.client = DataPipeClient()
                    init_ts = self.do_refresh(group, sfcs_conf_path)
                    if not init_ts:
                        raise RuntimeError(f'Credentials helper for {sfcs_conf_path} first fetch failed')
                    self.threads[group] = threading.Thread(
                        target=self.refresh_loop, args=(group, sfcs_conf_path, init_ts[0], init_ts[1]), daemon=True
                    )
                    self.threads[group].start()
                    self.running[group] = True
                    logger.info(f'CredentialsHelper refresh thread for {sfcs_conf_path} start')
                    return
        logger.info(f'CredentialsHelper thread for {sfcs_conf_path} is already running, do nothing')

    def stop(self):
        self.stop_flag.set()

    def is_valid_res(self, d: Optional[dict]) -> bool:
        if not d:
            return False
        for k in ['Cred', 'SfcsNameNodeAddress']:
            if k not in d:
                return False
        d = d['Cred']
        for k in ['CurrentTime', 'ExpiredTime', 'AccessKeyId', 'SecretAccessKey', 'SessionToken']:
            if k not in d:
                return False
        return True

    def refresh_loop(self, group: str, sfcs_conf_path: str, current_time: float, expired_time: float) -> None:
        while True:
            now = datetime.now(tz=timezone.utc).timestamp()
            ts_ref = (current_time + expired_time) / 2
            if now >= ts_ref:
                ts = self.do_refresh(group, sfcs_conf_path)
                if not ts:
                    raise RuntimeError(f'Credentials helper do refresh at {sfcs_conf_path} failed')
                current_time, expired_time = ts[0], ts[1]
            else:
                if self.stop_flag.wait(ts_ref - now):
                    return

    def do_refresh(self, group: str, sfcs_conf_path: str) -> Optional[Tuple[float, float]]:
        d = self.client.get_sfcs_ak_sk_st()
        if self.is_valid_res(d):
            name_node_ip = d['SfcsNameNodeAddress']
            d = d['Cred']
            current_time = datetime.fromisoformat(d['CurrentTime']).timestamp()
            expired_time = datetime.fromisoformat(d['ExpiredTime']).timestamp()
            ak = d['AccessKeyId']
            sk = d['SecretAccessKey']
            st = d['SessionToken']
            try:
                sfcs_fsname = json.loads(os.getenv('SFCS_FSNAME'))[group]
            except:
                sfcs_fsname = os.getenv('SFCS_FSNAME')

            logger.info(f"sfcs_fsname: {sfcs_fsname}")
            properties = init_sfcs_properties(group, use_datapipe=True)
            properties['cfs.access.key'] = ak
            properties['cfs.secret.key'] = sk
            properties['cfs.security.token'] = st
            properties['cfs.namenode.endpoint.address.' + sfcs_fsname] = name_node_ip
            generate_sfcs_conf_xml(sfcs_conf_path, properties)
            logger.info(f'Credentials are successfully refreshed at {sfcs_conf_path}!')
            return current_time, expired_time
        else:
            return None


credentials_helper = CredentialsHelper()


def init_sfcs_properties(group: str, use_datapipe: bool = False) -> dict:
    config: SfcsConfiguration = sfcs_default_config.override_configuration_with_env(group)
    try:
        config.validate(use_datapipe=use_datapipe)
    except Exception as e:
        logger.error(f"Validation failed for configuration: {config}. Original error: {e}")
        raise e
    properties = config.to_property()
    return properties


def build_uri(fs_name: str, region: str) -> str:
    return 'cfs://' + fs_name + '.sfcs-' + region + '.ivolces.com'


def build_metric_server_host(region: str) -> str:
    return 'metricserver-' + region + '.cfs.ivolces.com'


def path_mapper(file: FILE_PATH, policy_key: str) -> str:
    policy = sfcs_default_config.cache_policies.get(policy_key)
    if sfcs_default_config is None or policy is None or len(policy.cache_prefix) == 0:
        return file
    else:
        trimmed_path = file.removeprefix(policy.mount_path)
        trimmed_path = trimmed_path.removeprefix('/') if trimmed_path.startswith('/') else trimmed_path
        valid_path = os.path.join(policy.cache_prefix, trimmed_path)
        logger.info(f"path mapper: from {file} with prefix {policy.cache_prefix}, output: {valid_path}")
        return valid_path


def generate_sfcs_conf_xml(sfcs_conf: FILE_PATH, sfcs_properties: dict):
    doc = xml.dom.minidom.Document()
    configuration = doc.createElement('configuration')
    doc.appendChild(configuration)
    for key in sfcs_properties:
        property = doc.createElement('property')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(key))
        value = doc.createElement('value')
        value.appendChild(doc.createTextNode(sfcs_properties[key]))

        property.appendChild(name)
        property.appendChild(value)
        configuration.appendChild(property)

    pi = doc.createProcessingInstruction('xml-stylesheet', 'type="text/xsl" href="configuration.xsl"')
    doc.insertBefore(pi, configuration)

    tmp_conf = tempfile.NamedTemporaryFile(mode='w', delete=False)
    doc.writexml(tmp_conf, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    tmp_conf.close()
    shutil.move(tmp_conf.name, sfcs_conf)


def sfcs_conf_group(file: FILE_PATH) -> str:
    group_name = sfcs_default_config.get_mount_path_or_root_dir(path=file)
    return group_name


def generate_sfcs_conf_path(group: str) -> FILE_PATH:
    # the group name might be a/b/c,
    # therefore we concat as a#b#c
    group = group.removeprefix('/')
    valid_group = '#'.join(group.split("/"))
    return os.path.join(os.getcwd(), valid_group + '.xml')


def init_sfcs_conf(file: FILE_PATH):
    group = sfcs_conf_group(file)
    sfcs_conf = generate_sfcs_conf_path(group)
    os.environ['LIBCLOUDFS_CONF'] = sfcs_conf
    logger.info(f'environ LIBCLOUDFS_CONF set to {sfcs_conf}')

    if datapipe_needless():
        # case 1: env SFCS_ACCESS_KEY, SFCS_SECRET_KEY and SFCS_NAMENODE_ENDPOINT_ADDRESS exist
        logger.info('Use aksk and namenode_ip in env to generate sfcs config')
        properties = init_sfcs_properties(group)
        generate_sfcs_conf_xml(sfcs_conf, properties)
    else:
        # case 2: use datapipe socket to get and refresh ak, sk, and st
        logger.info('Use credentials helper to generate and update sfcs config')
        credentials_helper.run(group, sfcs_conf)
    return group


def datapipe_needless() -> bool:
    if os.getenv('SFCS_ACCESS_KEY') and os.getenv('SFCS_SECRET_KEY') and os.getenv('SFCS_NAMENODE_ENDPOINT_ADDRESS'):
        # case 1: for compatibility
        return True
    if sfcs_default_config.access_key and sfcs_default_config.secret_key:
        return True
    return False


def sfcs_get_file_size(file_path: str) -> int:
    sfcs_file = SFCSFile(file_path)
    return sfcs_file.get_file_size()


def sfcs_read_file(
    file_path: str,
    arr: np.ndarray,
    length: int,
    offset: int,
    num_thread: Optional[int] = 1,
    cipher_info: CipherInfo = CipherInfo(False),
) -> int:
    sfcs_file = SFCSFile(
        file_path,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )
    return sfcs_file.read_file_to_array(arr, length, offset, num_thread)


def sfcs_read_multi_files(
    file_paths: List[str],
    tensors: List[torch.Tensor],
    cipher_info: CipherInfo = CipherInfo(False),
):
    num_thread = len(file_paths)
    assert num_thread == len(tensors)
    lengths = [tensor.numel() * tensor.element_size() for tensor in tensors]
    offsets = [0 for _ in range(num_thread)]

    sfcs_fs = SFCSFs()
    sfcs_fs.read_multi_files(
        file_paths,
        tensors,
        lengths,
        offsets,
        num_thread,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )


def sfcs_write_file(
    file_path: str, arr: np.ndarray, length: int, cipher_info: CipherInfo = CipherInfo(False), append: bool = False
) -> int:
    sfcs_file = SFCSFile(
        file_path,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )
    return sfcs_file.write_file_from_array(arr, length, append)


def sfcs_write_multi_files(
    file_paths: List[str],
    tensors: List[torch.Tensor],
    cipher_info: CipherInfo = CipherInfo(False),
):
    num_thread = len(file_paths)
    assert num_thread == len(tensors)
    lengths = [tensor.numel() * tensor.element_size() for tensor in tensors]
    offsets = [0 for _ in range(num_thread)]

    sfcs_fs = SFCSFs()
    sfcs_fs.write_multi_files(
        file_paths,
        tensors,
        lengths,
        offsets,
        num_thread,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )


def sfcs_get_multi_file_size(
    file_paths: List[str],
):
    num_thread = len(file_paths)
    lengths = []

    sfcs_fs = SFCSFs()
    sfcs_fs.get_multi_file_size(file_paths, lengths, num_thread)
    return lengths


def sfcs_write_file_in_parallel(
    file_path: str,
    tensors: List[torch.Tensor],
    sizes: List[int],
    offsets: List[int],
    st_header_bytes,
    st_header_len,
    cipher_info: CipherInfo = CipherInfo(False),
):
    sfcs_delete_file(file_path)
    group = sfcs_conf_group(file_path)
    try:
        sfcs_concat_dir = json.loads(os.getenv('SFCS_CONCAT_DIR'))[group]
    except:
        sfcs_concat_dir = os.getenv('SFCS_CONCAT_DIR')
    tmp_concat_dir = os.path.join(sfcs_concat_dir, str(hash(file_path)))
    sfcs_delete_file(tmp_concat_dir)
    sfcs_mkdir(tmp_concat_dir)
    concat_file_path = os.path.join(tmp_concat_dir, "concat")

    if cipher_info.use_cipher and cipher_info.use_header:
        h_off = CipherInfo.HEADER_SIZE
        file_bytes = np.empty(st_header_len + h_off, dtype=np.byte)
        file_bytes[:h_off] = np.frombuffer(cipher_info.to_header_bytes(), dtype=np.byte)
        file_bytes[h_off:] = np.frombuffer(st_header_bytes, dtype=np.byte)
    else:
        file_bytes = np.frombuffer(st_header_bytes, dtype=np.byte)
    sfcs_write_file(concat_file_path, file_bytes, len(file_bytes), cipher_info)

    raw_tensors = []
    for tensor in tensors:
        raw_tensors.append(tensor.contiguous())

    file_offsets = []
    header_size = sfcs_get_file_size(concat_file_path)
    for offset in offsets:
        file_offsets.append(offset + header_size)

    sfcs_file = SFCSFile(
        file_path,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )
    sfcs_file.write_file_from_tensors(raw_tensors, sizes, file_offsets, tmp_concat_dir, concat_file_path)
    sfcs_delete_file(tmp_concat_dir)


def sfcs_delete_file(file_path: str):
    sfcs_file = SFCSFile(file_path)
    sfcs_file.delete_file()


def sfcs_mkdir(file_path: str):
    sfcs_fs = SFCSFs()
    sfcs_fs.mkdir(file_path)
