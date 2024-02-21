# veTurboIO

火山引擎研发的一款用于高性能读写 PyTorch 模型文件的 Python 库。该库实现了主要基于 safetensors 文件格式，实现高效的存储与读取张量数据。

## 安装

```bash
cd veturboio
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

### 转换现有的 PyTorch 文件
```bash
python -m veturboio.convert -i model.pt -o model.safetensors
```


## 特性

- 多线程读取文件；
- zero-copy 读取，不额外花费内存；
- 支持直接加载到 CUDA；
- BFloat16 数值支持；
- 固定 pinmem 用于快速反复读取；
- 兼容 PyTorch 标准格式（无性能提升）；
- 兼容 safetensors 格式；

## 收益

标准的 PyTorch 模型文件会经过 zip 与 pickle 两次操作，这两个操作极大的抑制了读取的速度，同时 unpickle 也会带来潜在的不安全性。我们使用一种自定义的模型格式来存储 tensor 数据，希望可以改善 PyTorch 标准格式所存在的这些问题。目前已经实现的优点有：

- 多线程读取：当前文件对象主要的存放点为云端存储，单一进程无法达到云存储的带宽上限，必须使用多线程读取才能达到最大的读取速度。PyTorch 标准格式的读取速度受限于 pickle 解析速度，远无法达到云存储的速度上限；
- 云端适配：基于火山引擎的云端存储（vePFS、SFCS）特性，最大化的利用了云端存储的带宽；
- 安全性：不再使用 pickle 对象，避免了 pickle 的安全性问题；

