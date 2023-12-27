# 加解密模型文件

该库底层通过两种接口读写：SFCS SDK 和 POSIX。如果文件路径前缀为 `sfcs://` 就视为使用 SFCS SDK，所需的鉴权信息可以从火山引擎可信服务的 `unix domain socket` 获取或者设置以下三个环境变量：

| 环境变量名                     | 含义                              |
| ------------------------------ | --------------------------------- |
| SFCS_ACCESS_KEY                | SFCS 文件系统的 AK                  |
| SFCS_SECRET_KEY                | SFCS 文件系统的 SK                  |
| SFCS_NAMENODE_ENDPOINT_ADDRESS | SFCS 文件系统 NameNode 地址          |


加解密读写模型文件需要 data key 和 iv，有 3 种获取方式，读取优先级按照下列顺序：
- [1] 加密的 data key 和 iv 存放在密文模型文件的 header 中，使用火山引擎 KMS 解密得到明文的 data key。
- [1.1] 访问 KMS 所需的 AK/SK/ST 从火山引擎可信服务的 unix domain socket 获取，需要额外挂载。
- [1.2] 访问 KMS 所需的 AK/SK/ST 从环境变量获取。
- [2] 访问火山引擎可信服务的 unix domain socket 直接获取 data key 和 iv，需要额外挂载。
- [3] 通过环境变量直接设置 data key 和 iv。

不同方式需要设置的环境变量如下：

| 环境变量名                     | 含义                                 |
| ------------------------------ | --------------------------------- |
| VETURBOIO_KMS_HOST             |  [1] KMS 服务地址，默认值 open.volcengineapi.com|
| VETURBOIO_KMS_REGION            | [1] KMS 服务所在区域，默认值 cn-beijing |
| VETURBOIO_KMS_KEYRING_NAME      | [1] KMS 服务解密 data key 的钥匙环名 |
| VETURBOIO_KMS_KEY_NAME          | [1] KMS 服务解密 data key 的主密钥名 |
| DATAPIPE_SOCKET_PATH            | [1.1][2] 可信服务 uds 的路径        |
| VETURBOIO_KMS_ACCESS_KEY        | [1.2] KMS 鉴权的 AK |
| VETURBOIO_KMS_SECRET_KEY        | [1.2] KMS 鉴权的 SK |
| VETURBOIO_KMS_SESSION_TOKEN     | [1.2] KMS 鉴权的临时令牌，非必需|
| VETURBOIO_KEY                   | [3] 加解密的 128 位数据密钥的 base64 编码 |
| VETURBOIO_IV                    | [3] 加解密的 128 位初始向量的 base64 编码 |


按照上述三种方式设置好后，可以参考下面代码在读写模型文件时启用加解密：
```python
import torch
import veturboio

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}

# use cpu to encrypt
veturboio.save_file(tensors, "sfcs://model.safetensors", use_cipher=True)

# use cpu to decrypt if map_location is cpu
reloaded_tensor1 = veturboio.load("sfcs://model.safetensors", map_location="cpu", use_cipher=True)

# use gpu to decrypt if map_location is cuda
reloaded_tensor2 = veturboio.load("sfcs://model.safetensors", map_location="cuda:0", use_cipher=True)

# check if the tensors are the same
for k, v in tensors.items():
    assert torch.allclose(v, reloaded_tensor1[k])
for k, v in tensors.items():
    assert torch.allclose(v, reloaded_tensor2[k])
```

