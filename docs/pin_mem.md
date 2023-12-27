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

