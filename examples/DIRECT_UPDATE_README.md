# 直接使用已有的输入输出和reward更新参数

如果你已经有了模型的输入（prompt）、输出（response）和对应的reward，可以直接使用这些数据更新模型参数，而不需要重新rollout。

## 核心步骤

### 1. 准备数据

你需要准备以下数据：
- **prompts**: 输入提示列表
- **responses**: 模型输出列表  
- **rewards**: 对应的reward列表（标量值）

可选：
- **ref_log_probs**: 参考策略的log_probs（用于计算KL散度）

### 2. 数据格式转换

将数据转换为 `DataProto` 格式：

```python
from verl.protocol import DataProto
from tensordict import TensorDict

# Tokenize
prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
response_tokens = tokenizer(responses, return_tensors="pt", padding=True)

# 合并input_ids
input_ids = torch.cat([prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1)
attention_mask = torch.cat([prompt_tokens["attention_mask"], response_tokens["attention_mask"]], dim=1)

# 计算advantages（基于reward）
rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
baseline = rewards_tensor.mean()
advantages = (rewards_tensor - baseline).unsqueeze(-1).expand(-1, response_length)
advantages = advantages * response_mask.float()

# 创建DataProto
batch_data = TensorDict({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "responses": response_tokens["input_ids"],
    "advantages": advantages,
    "old_log_probs": torch.zeros(batch_size, response_length),  # 占位符
}, batch_size=[batch_size])

dataproto = DataProto(
    batch=batch_data,
    non_tensor_batch={},
    meta_info={"temperature": 1.0}
)
```

### 3. 计算old_log_probs

使用当前模型计算old_log_probs：

```python
from verl.workers.actor.dp_actor import DataParallelPPOActor

actor = DataParallelPPOActor(
    config=actor_config,
    actor_module=model,
    actor_optimizer=optimizer,
)

with torch.no_grad():
    old_log_probs = actor.compute_log_prob(dataproto)
    dataproto.batch["old_log_probs"] = old_log_probs
```

### 4. 更新参数

调用 `update_policy` 方法更新参数：

```python
metrics = actor.update_policy(dataproto)
```

## 完整示例

参考 `examples/direct_update_example.py` 查看完整示例代码。

## 关键点

1. **old_log_probs**: 如果数据是从已有模型生成的，需要保存当时的log_probs。如果是新数据，使用当前模型计算。

2. **advantages计算**: 
   - 简单方法：`advantage = reward - baseline`
   - GAE方法：需要values，使用 `core_algos.compute_gae_advantage_return()`
   - 其他方法：GRPO、RLOO等，参考 `core_algos.py`

3. **数据格式**: 必须使用 `DataProto` 和 `TensorDict` 格式，这是框架的要求。

4. **配置**: 需要正确设置 `ActorConfig`，特别是：
   - `global_batch_size_per_device`
   - `micro_batch_size_per_device_for_update`
   - `ppo_epochs`
   - `clip_ratio_low` / `clip_ratio_high`

## 注意事项

- 如果数据是从旧模型生成的，建议保存 `old_log_probs`，这样更准确
- 如果使用KL散度损失，需要提供 `ref_log_probs`
- advantages的计算方法会影响训练效果，建议根据实际情况选择

