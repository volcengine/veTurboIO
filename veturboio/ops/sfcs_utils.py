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

import json
import os
import shutil
import tempfile
import threading
import xml.dom.minidom
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from veturboio.ops.cipher import CipherInfo, DataPipeClient
from veturboio.types import FILE_PATH

try:
    from veturboio.utils.load_veturboio_ext import load_veturboio_ext

    veturboio_ext = load_veturboio_ext()
    SFCSFile = veturboio_ext.SFCSFile
except ImportError:
    SFCSFile = None
    logger.warning("veturboio_ext not found, fallback to pure python implementation")

SFCS_REQ_ENV_LIST = [
    'SFCS_FSNAME',
    'SFCS_REGION',
    'SFCS_AUTHENTICATION_SERVICE_NAME',
    'SFCS_NS_ID',
    'SFCS_UFS_PATH',
    'SFCS_MULTI_NIC_WHITELIST',
    'SFCS_NETWORK_SEGMENT',
    'SFCS_LOG_SEVERITY',
]

SFCS_OPT_ENV_LIST = [
    'SFCS_ACCESS_KEY',
    'SFCS_SECRET_KEY',
    'SFCS_NAMENODE_ENDPOINT_ADDRESS',
]


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
            properties = init_sfcs_properties(group)
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


def init_sfcs_properties(group: str) -> dict:
    for env in SFCS_REQ_ENV_LIST:
        if os.getenv(env) is None:
            raise ValueError('environ ' + env + ' not set')

    try:
        sfcs_fsname = json.loads(os.getenv('SFCS_FSNAME'))[group]
        sfcs_ns_id = json.loads(os.getenv('SFCS_NS_ID'))[group]
        sfcs_ufs_path = json.loads(os.getenv('SFCS_UFS_PATH'))[group]
        logger.info(f"parse sfcs fsname, ns_id and ufs_path from environ in JSON format")
    except:
        sfcs_fsname = os.getenv('SFCS_FSNAME')
        sfcs_ns_id = os.getenv('SFCS_NS_ID')
        sfcs_ufs_path = os.getenv('SFCS_UFS_PATH')
        logger.info(f"parse sfcs fsname, ns_id and ufs_path from environ in STRING format")

    properties = default_sfcs_properties()
    properties['dfs.default.uri'] = 'cfs://' + sfcs_fsname + '.sfcs-' + os.getenv('SFCS_REGION') + '.ivolces.com'
    properties['dfs.authentication.service.name'] = os.getenv('SFCS_AUTHENTICATION_SERVICE_NAME')
    properties['cfs.filesystem.ns-id'] = sfcs_ns_id
    properties['cfs.filesystem.ufs-path'] = sfcs_ufs_path
    properties['cfs.metrics.server.host'] = 'metricserver.cfs-' + os.getenv('SFCS_REGION') + '.ivolces.com'
    properties['cfs.client.multi-nic.whitelist'] = os.getenv('SFCS_MULTI_NIC_WHITELIST')
    properties['cfs.client.network.segment'] = os.getenv('SFCS_NETWORK_SEGMENT')
    properties['dfs.client.log.severity'] = os.getenv('SFCS_LOG_SEVERITY')

    # optional
    properties['cfs.filesystem.task-id'] = os.getenv("SFCS_TASK_ID", "sfcs")
    properties['cfs.filesystem.sync-interval'] = os.getenv('SFCS_SYNC_INTERVAL', "-1")
    properties['cfs.access.key'] = os.getenv('SFCS_ACCESS_KEY')
    properties['cfs.secret.key'] = os.getenv('SFCS_SECRET_KEY')
    properties['cfs.namenode.endpoint.address.' + sfcs_fsname] = os.getenv('SFCS_NAMENODE_ENDPOINT_ADDRESS')
    return properties


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
    return os.path.abspath(file).split("/")[1]


def init_sfcs_conf(file: FILE_PATH):
    group = sfcs_conf_group(file)
    sfcs_conf = os.path.join(os.getcwd(), group + '.xml')
    os.environ['LIBCFS_CONF'] = sfcs_conf
    logger.info(f'environ LIBCFS_CONF set to {sfcs_conf}')

    if os.path.isfile(sfcs_conf):
        # case 1: a xml file already exists, do nothing
        logger.info('LIBCFS_CONF file exists')
    else:
        if (
            os.getenv('SFCS_ACCESS_KEY')
            and os.getenv('SFCS_SECRET_KEY')
            and os.getenv('SFCS_NAMENODE_ENDPOINT_ADDRESS')
        ):
            # case 2: env SFCS_ACCESS_KEY, SFCS_SECRET_KEY and SFCS_NAMENODE_ENDPOINT_ADDRESS exist
            logger.info('Use aksk and namenode_ip in env to generate sfcs config')
            properties = init_sfcs_properties(group)
            generate_sfcs_conf_xml(sfcs_conf, properties)
        else:
            # case 3: use datapipe socket to get and refresh ak, sk, and st
            logger.info('Use credentials helper to generate and update sfcs config')
            credentials_helper.run(group, sfcs_conf)


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


def sfcs_write_file(file_path: str, arr: np.ndarray, length: int, cipher_info: CipherInfo = CipherInfo(False)) -> int:
    sfcs_file = SFCSFile(
        file_path,
        cipher_info.use_cipher,
        cipher_info.key,
        cipher_info.iv,
        CipherInfo.HEADER_SIZE if cipher_info.use_header else 0,
    )
    return sfcs_file.write_file_from_array(arr, length)


def sfcs_delete_file(file_path: str):
    sfcs_file = SFCSFile(file_path)
    sfcs_file.delete_file()
