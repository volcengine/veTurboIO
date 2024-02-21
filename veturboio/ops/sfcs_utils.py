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
import shutil
import tempfile
import threading
import xml.dom.minidom
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger

from veturboio.ops.cipher import CipherInfo, DataPipeClient

try:
    import veturboio_ext

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

SFCS_PROPERTIES = {
    'cfs.filesystem.fs-mode': 'ACC',
    'cfs.filesystem.task-id': 'sfcs',
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
        self.running = False
        # daemon thread will stop when parent thread exits
        self.thread = None
        self.client = None
        self.current_time = 0
        self.expired_time = 0
        self.ak = None
        self.sk = None
        self.st = None
        self.name_node_ip = None
        self.sfcs_conf_path = None
        self.stop_flag = None

    def run(self, sfcs_conf_path) -> None:
        if not self.running:
            with self.lock:
                if not self.running:
                    self.thread = threading.Thread(target=self.refresh_loop, daemon=True)
                    self.stop_flag = threading.Event()
                    self.client = DataPipeClient()
                    if not self.client.session:
                        raise RuntimeError('Datapipe client initialization failed in credentials helper')
                    self.sfcs_conf_path = sfcs_conf_path
                    if not self.do_refresh():
                        raise RuntimeError('Credentials helper do refresh failed')
                    self.thread.start()
                    self.running = True
                    logger.info('CredentialsHelper refresh thread strat')
                    return
        logger.info('CredentialsHelper thread is already running, do nothing')

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

    def refresh_loop(self) -> None:
        while True:
            now = datetime.now(tz=timezone.utc).timestamp()
            ts_ref = (self.current_time + self.expired_time) / 2
            if now >= ts_ref:
                if not self.do_refresh():
                    raise RuntimeError('Credentials helper do refresh failed')
            else:
                if self.stop_flag.wait(ts_ref - now):
                    return

    def do_refresh(self) -> bool:
        d = self.client.get_sfcs_ak_sk_st()
        if self.is_valid_res(d):
            self.name_node_ip = d['SfcsNameNodeAddress']
            d = d['Cred']
            self.current_time = datetime.fromisoformat(d['CurrentTime']).timestamp()
            self.expired_time = datetime.fromisoformat(d['ExpiredTime']).timestamp()
            self.ak = d['AccessKeyId']
            self.sk = d['SecretAccessKey']
            self.st = d['SessionToken']
            # update SFCS_PROPERTIES and then write xml
            SFCS_PROPERTIES['cfs.access.key'] = self.ak
            SFCS_PROPERTIES['cfs.secret.key'] = self.sk
            SFCS_PROPERTIES['cfs.security.token'] = self.st
            SFCS_PROPERTIES['cfs.namenode.endpoint.address.' + os.getenv('SFCS_FSNAME')] = self.name_node_ip
            generate_sfcs_conf_xml(self.sfcs_conf_path)
            logger.info('Credentials are successfully refreshed!')
            return True
        else:
            return False


credentials_helper = CredentialsHelper()


def init_sfcs_properties():
    for env in SFCS_REQ_ENV_LIST:
        if os.getenv(env) is None:
            raise ValueError('environ ' + env + ' not set')

    SFCS_PROPERTIES['dfs.default.uri'] = (
        'cfs://' + os.getenv('SFCS_FSNAME') + '.sfcs-' + os.getenv('SFCS_REGION') + '.ivolces.com'
    )
    SFCS_PROPERTIES['dfs.authentication.service.name'] = os.getenv('SFCS_AUTHENTICATION_SERVICE_NAME')
    SFCS_PROPERTIES['cfs.filesystem.ns-id'] = os.getenv('SFCS_NS_ID')
    SFCS_PROPERTIES['cfs.filesystem.ufs-path'] = os.getenv('SFCS_UFS_PATH')
    SFCS_PROPERTIES['cfs.metrics.server.host'] = 'metricserver.cfs-' + os.getenv('SFCS_REGION') + '.ivolces.com'
    SFCS_PROPERTIES['cfs.client.multi-nic.whitelist'] = os.getenv('SFCS_MULTI_NIC_WHITELIST')
    SFCS_PROPERTIES['cfs.client.network.segment'] = os.getenv('SFCS_NETWORK_SEGMENT')
    SFCS_PROPERTIES['dfs.client.log.severity'] = os.getenv('SFCS_LOG_SEVERITY')

    # optional
    SFCS_PROPERTIES['cfs.filesystem.sync-interval'] = os.getenv('SFCS_SYNC_INTERVAL', "-1")
    SFCS_PROPERTIES['cfs.access.key'] = os.getenv('SFCS_ACCESS_KEY')
    SFCS_PROPERTIES['cfs.secret.key'] = os.getenv('SFCS_SECRET_KEY')
    SFCS_PROPERTIES['cfs.namenode.endpoint.address.' + os.getenv('SFCS_FSNAME')] = os.getenv(
        'SFCS_NAMENODE_ENDPOINT_ADDRESS'
    )


def generate_sfcs_conf_xml(sfcs_conf):
    doc = xml.dom.minidom.Document()
    configuration = doc.createElement('configuration')
    doc.appendChild(configuration)
    for key in SFCS_PROPERTIES:
        property = doc.createElement('property')
        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(key))
        value = doc.createElement('value')
        value.appendChild(doc.createTextNode(SFCS_PROPERTIES[key]))

        property.appendChild(name)
        property.appendChild(value)
        configuration.appendChild(property)

    pi = doc.createProcessingInstruction('xml-stylesheet', 'type="text/xsl" href="configuration.xsl"')
    doc.insertBefore(pi, configuration)

    tmp_conf = tempfile.NamedTemporaryFile(mode='w', delete=False)
    doc.writexml(tmp_conf, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    tmp_conf.close()
    shutil.move(tmp_conf.name, sfcs_conf)


def init_sfcs_conf():
    if not os.getenv('LIBCFS_CONF'):
        logger.warning('environ LIBCFS_CONF not set, set it to ' + os.getcwd() + '/libcfs.xml')
        os.environ['LIBCFS_CONF'] = os.getcwd() + '/libcfs.xml'

    sfcs_conf = os.getenv('LIBCFS_CONF')
    if os.path.exists(sfcs_conf):
        # case 1: a xml file already exists, do nothing
        logger.warning('LIBCFS_CONF file exists')
    else:
        init_sfcs_properties()
        if (
            os.getenv('SFCS_ACCESS_KEY')
            and os.getenv('SFCS_SECRET_KEY')
            and os.getenv('SFCS_NAMENODE_ENDPOINT_ADDRESS')
        ):
            # case 2: env SFCS_ACCESS_KEY, SFCS_SECRET_KEY and SFCS_NAMENODE_ENDPOINT_ADDRESS exist
            logger.warning('Use aksk and namenode_ip in env to generate sfcs config')
            generate_sfcs_conf_xml(sfcs_conf)
        else:
            # case 3: use datapipe socket to get and refresh ak, sk, and st
            logger.warning('Use credentials helper to generate and update sfcs config')
            credentials_helper.run(sfcs_conf)


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
    sfcs_file = SFCSFile(file_path, cipher_info.use_cipher, cipher_info.key, cipher_info.iv)
    return sfcs_file.read_file_to_array(arr, length, offset, num_thread)


def sfcs_write_file(file_path: str, arr: np.ndarray, length: int, cipher_info: CipherInfo = CipherInfo(False)) -> int:
    sfcs_file = SFCSFile(file_path, cipher_info.use_cipher, cipher_info.key, cipher_info.iv)
    return sfcs_file.write_file_from_array(arr, length)


def sfcs_delete_file(file_path: str):
    sfcs_file = SFCSFile(file_path)
    sfcs_file.delete_file()
