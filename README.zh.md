# veTurboIO


[en](./README.md) | [中文](./README.zh.md)


一个由 Volcano Engine 开发的用于高性能读写 PyTorch 模型文件的 Python 库。该库主要基于 safetensors 文件格式实现，以实现对张量数据的高效存储和读取。

## 安装

可以直接通过以下方式安装：
```bash
pip install veturboio -f https://veturbo-cn-beijing.tos-cn-beijing.volces.com/veturboio/index.html --no-build-isolation
```

提示：此指令会优先下载与当前 Python 和 PyTorch 版本匹配的 whl 文件，如果没有找到匹配的 whl 文件，会自动下载源码进行编译安装。

如果安装失败，也可以尝试通过下载源码安装，然后手动编译安装。
```bash
# CUDA ops, default
python setup.py install --cuda_ext

# NPU ops
python setup.py install --npu_ext

# CPU only
python setup.py install --cpu_ext
```

## 快速开始

### 读写模型文件


```python
import torch
import veturboio

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}

veturboio.save_file(tensors, "model.safetensors")

new_tensors = veturboio.load("model.safetensors")

# check if the tensors are the same
for k, v in tensors.items():
    assert torch.allclose(v, new_tensors[k])
```

## 转换已有 PyTorch 文件

```bash
python -m veturboio.convert -i model.pt -o model.safetensors
```

## 性能测试

直接运行：
```bash
bash bench/io_bench.sh
```

接下来，你可以获得如下的结果：
```
fs_name    tensor_size     veturboio load_time(s)             torch load_time(s)
shm        1073741824      0.08                               0.63
shm        2147483648      0.19                               1.26
shm        4294967296      0.36                               2.32
```

## 进阶功能

### 使用 veMLP 加速读写
Volcano Engine Machine Learning Platform (veMLP) 提供了基于 GPU 集群的物理磁盘的分布式缓存文件系统。

<p align="center">
    <img src="./docs/imgs/SFCS.png" style="zoom:15%;">
</p>

当集群级任务需要读取模型文件时，缓存系统可以通过 RDMA 传输高效地在 GPU 机器之间分发模型文件，从而避免网络传输瓶颈。使用此系统时，veTurboIO 可以最大化其性能优势。


### 加密和解密模型文件

veTurboIO 支持模型文件的加密和解密。您可以阅读[教程]([tutorial](./docs/encrypt_model.md))以了解如何保护您的模型文件。当您使用 GPU 作为目标设备时，veTurboIO 可以实时解密模型文件。

## 许可证

[Apache License 2.0](./LICENSE)
