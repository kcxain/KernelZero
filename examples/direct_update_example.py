#!/usr/bin/env python3
"""
直接使用已有的模型输入输出和reward更新参数的示例

核心步骤：
1. 准备数据：输入(prompt)、输出(response)、reward
2. 计算old_log_probs（如果需要）
3. 计算advantages（基于reward）
4. 调用actor.update_policy()更新参数
"""

import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from verl.protocol import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.config import ActorConfig
from verl.trainer import core_algos


def direct_update_example():
    """
    示例：如何使用已有的输入输出和reward直接更新参数
    """
    # 1. 加载模型和tokenizer
    model_path = "/path/to/your/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 2. 准备数据（假设你已经有了这些数据）
    prompts = [
        "问题：什么是PyTorch？",
        "问题：如何使用torch.nn.Linear？",
    ]
    responses = [
        "PyTorch是一个深度学习框架...",
        "torch.nn.Linear是线性层...",
    ]
    rewards = [0.8, 0.6]  # 对应的reward
    
    # 3. Tokenize数据
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    response_tokens = tokenizer(responses, return_tensors="pt", padding=True, truncation=True)
    
    batch_size = len(prompts)
    input_ids = torch.cat([prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1)
    attention_mask = torch.cat([prompt_tokens["attention_mask"], response_tokens["attention_mask"]], dim=1)
    
    # 创建position_ids
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    response_length = response_tokens["input_ids"].shape[1]
    response_mask = attention_mask[:, -response_length:]
    
    # 4. 计算advantages（基于reward）
    # 方法1：简单REINFORCE
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    baseline = rewards_tensor.mean()
    advantages = (rewards_tensor - baseline).unsqueeze(-1).expand(-1, response_length)
    advantages = advantages * response_mask.float()
    
    # 方法2：如果需要使用GAE，需要先计算values（这里简化处理）
    # advantages, returns = core_algos.compute_gae_advantage_return(
    #     token_level_rewards=token_level_rewards,
    #     values=values,
    #     response_mask=response_mask,
    #     gamma=torch.tensor(1.0),
    #     lam=torch.tensor(0.95),
    # )
    
    # 5. 创建DataProto
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
    
    # 6. 创建Actor和优化器
    actor_config = ActorConfig(
        global_batch_size_per_device=batch_size,
        micro_batch_size_per_device_for_update=batch_size,
        micro_batch_size_per_device_for_experience=batch_size,
        ppo_epochs=1,
        temperature=1.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    actor = DataParallelPPOActor(
        config=actor_config,
        actor_module=model,
        actor_optimizer=optimizer,
    )
    
    # 7. 计算old_log_probs（使用当前模型）
    with torch.no_grad():
        old_log_probs = actor.compute_log_prob(dataproto)
        dataproto.batch["old_log_probs"] = old_log_probs
    
    # 8. 更新参数
    metrics = actor.update_policy(dataproto)
    
    print("更新完成！指标：")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"  {key}: {sum(value) / len(value):.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    return metrics


def batch_update_example(data_list: List[Dict]):
    """
    批量更新示例
    
    Args:
        data_list: 包含多个样本的列表，每个样本包含：
            - prompt: 输入文本
            - response: 输出文本
            - reward: reward值
            - ref_log_probs: 可选，参考策略的log_probs
    """
    # 提取数据
    prompts = [item["prompt"] for item in data_list]
    responses = [item["response"] for item in data_list]
    rewards = [item["reward"] for item in data_list]
    ref_log_probs_list = [item.get("ref_log_probs") for item in data_list]
    
    # ... 后续步骤与direct_update_example相同
    pass


if __name__ == "__main__":
    # 注意：这只是一个示例，实际使用时需要：
    # 1. 替换模型路径
    # 2. 准备真实的数据
    # 3. 根据实际情况调整配置
    
    print("这是一个示例脚本，展示如何使用已有的输入输出和reward更新参数")
    print("请根据实际情况修改代码")
    
    # direct_update_example()

