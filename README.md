# 1 配置示例

## 1.1 .env 配置

```text
CONFIG_PATH=/root/***/***/config.json

TORCH_CUDA_ARCH_LIST=*.*

VLLM_CACHE_ROOT=/root/***/.cache/
```

# 1.2 config.json 配置

参考配置

> **common** 配置项待定

> **models** 请按照模型名称分类，并且每一个模型下配置一个数组，表示实际的实例：
> - **devices**：配置实例的 GPU 设备号，以及权重，权重越大使用概率越大，必须是大于零的整数或者浮点数
> - **engine_args**：配置引擎参数，参考 https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
>

```json
{
  "common": {
  },
  "models": {
    "Qwen/Qwen2.5-32B-Instruct": [
      {
        "devices": {
          "gpu_ids": [0, 1],
          "weight": 1
        },
        "engine_args": {
          "model": "/root/***/Qwen2.5-32B-Instruct/",
          "max_model_len": 8192,
          "max_num_seqs": 256,
          "max_num_batched_tokens": 16384,
          "gpu_memory_utilization": 0.90,
          "tensor_parallel_size": 1
        }
      },
      {
        "devices": {
          "gpu_ids": [3, 4],
          "weight": 2
        },
        "engine_args": {
          "model": "/root/***/Qwen2.5-32B-Instruct/",
          "max_model_len": 8192,
          "max_num_seqs": 256,
          "max_num_batched_tokens": 16384,
          "gpu_memory_utilization": 0.90,
          "tensor_parallel_size": 1
        }
      }
    ]
  }
}
```

# 2 启动

程序主入口为根目录下的 **main.py**

测试可以使用 **tests.ipynb**

# 3 参考

参考文档 : https://www.openaidoc.com.cn/api-reference/completions