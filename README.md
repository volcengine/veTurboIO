# veTurboIO

火山引擎研发的一款用于高性能读写 PyTorch 模型文件的 Python 库。该库实现了主要基于 safetensors 文件格式，实现高效的存储与读取张量数据。

## 安装

可以直接通过以下方式进行安装：
```bash
pip install veturboio -f https://veturbo-cn-beijing.tos-cn-beijing.volces.com/veturboio/index.html
```

Tips: 该指令会优先下载与当前 Python、PyTorch 版本匹配的 whl 文件，如果没有找到匹配的 whl 文件，会自动下载源码进行编译安装。
当使用源码安装时，可增加 `--no-build-isolation` 来使用当前的运行环境进行编译并安装（否则会尝试创建虚拟环境）。


如果已经安装失败，可以尝试通过下载源码进行安装：
```bash
cd veturboio
python setup.py get_libcfs
python setup.py install
```

## 快速开始

```python
import torch
import veturboio

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}

veturboio.save_file(tensors, "model.safetensors")

reloaded_tensor = veturboio.load("model.safetensors", map_location="cpu")

# check if the tensors are the same
for k, v in tensors.items():
    assert torch.allclose(v, reloaded_tensor[k])
```

### 使用锁页内存加速连续加载数据到GPU
```python
import torch
import veturboio

tensors1 = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}

veturboio.save_file(tensors1, "model1.safetensors")

tensors2 = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}

veturboio.save_file(tensors2, "model2.safetensors")

helper = veturboio.init_io_helper()
reloaded_tensor1 = veturboio.load("model1.safetensors", map_location="cuda:0", use_pinmem=True, helper=helper)
# the map_location may be different
reloaded_tensor2 = veturboio.load("model2.safetensors", map_location="cuda:0", use_pinmem=True, helper=helper) 

# check if the tensors are the same
for k, v in tensors1.items():
    assert torch.allclose(v.cuda(), reloaded_tensor1[k])
for k, v in tensors2.items():
    assert torch.allclose(v.cuda(), reloaded_tensor2[k])
```

### 使用SFCS读写模型并启用加解密
该库可以读写 SFCS 上的模型文件，在此情况下可以启用加解密能力。读写 SFCS 上的模型文件和加解密所需的敏感信息，有两种获取方式：(1) 火山引擎可信服务的 unix domain socket，(2) 环境变量。在没有挂载可信服务 uds 的情况下，可以使用下面的环境变量：

| 环境变量名                     | 含义                              |
| ------------------------------ | --------------------------------- |
| SFCS_ACCESS_KEY                | SFCS文件系统的AK                  |
| SFCS_SECRET_KEY                | SFCS文件系统的SK                  |
| SFCS_NAMENODE_ENDPOINT_ADDRESS | SFCS文件系统name节点地址          |
| VETUROIO_KEY                   | 加解密的128位数据密钥的base64编码 |
| VETUROIO_IV                    | 加解密的128位初始向量的base64编码 |

挂载可信服务 uds 或者配置好环境变量后，可以参考下面代码在读写 SFCS 上的模型文件时启用加解密：
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

### 转换现有的 PyTorch 文件
```bash
python -m veturboio.convert -i model.pt -o model.safetensors
```

## 性能测试
直接运行
```bash
bash bench/io_bench.sh
```
可以得到如下结果
```
fs_name    tensor_size     veturboio load_time(s)             torch load_time(s)            
shm        1073741824      0.08                               0.63                              
shm        2147483648      0.19                               1.26                              
shm        4294967296      0.36                               2.32    
```
也可以进一步根据以下命令的参数说明调整使用参数
```bash
python bench/io_bench.py -h
```

## 特性

- [x] 多线程高性能读取文件；
- [x] zero-copy 读取，不额外花费内存；
- [x] 支持直接加载到 CUDA；
- [x] bfloat16 数值 类型支持；
- [x] 支持固定 pin-memory 用于让 GPU 快速反复读取大文件；
- [x] 兼容 PyTorch 标准格式（无性能提升）；
- [x] 兼容 safetensors 格式；
- [x] 特殊加密格式存储；

## 收益

标准的 PyTorch 模型文件会经过 zip 与 pickle 两次操作，这两个操作极大的抑制了读取的速度，同时 unpickle 也会带来潜在的不安全性。我们使用一种自定义的模型格式来存储 tensor 数据，希望可以改善 PyTorch 标准格式所存在的这些问题。目前已经实现的优点有：

- 多线程读取：当前文件对象主要的存放点为云端存储，单一进程无法达到云存储的带宽上限，必须使用多线程读取才能达到最大的读取速度。PyTorch 标准格式的读取速度受限于 pickle 解析速度，远无法达到云存储的速度上限；
- 云端适配：基于火山引擎的云端存储（vePFS、SFCS）特性，最大化的利用了云端存储的带宽；
- 安全性：不再使用 pickle 对象，避免了 pickle 的安全性问题；

## 更新记录

前往 [CHANGELOG](./CHANGELOG.md) 了解更多。