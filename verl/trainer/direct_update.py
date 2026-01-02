#!/usr/bin/env python3
"""
直接使用已有的模型输入输出和reward更新参数

使用场景：
- 你已经有了模型的输入（prompt）、输出（response）和对应的reward
- 想要直接更新模型参数，而不需要重新rollout

示例用法：
    python direct_update.py \
        --model_path /path/to/model \
        --input_file /path/to/inputs_outputs_rewards.json \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..protocol import DataProto
from ..workers.actor.dp_actor import DataParallelPPOActor
from ..workers.actor.config import ActorConfig
from ..trainer import core_algos
from ..utils import torch_functional as VF


def load_data_from_file(file_path: str) -> List[Dict]:
    """
    从文件加载数据
    
    期望的JSON格式：
    [
        {
            "prompt": "输入文本",
            "response": "模型输出文本",
            "reward": 0.5,  # 标量reward
            "ref_log_probs": [...]  # 可选：参考策略的log_probs
        },
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_dataproto(
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    tokenizer,
    ref_log_probs: Optional[List[torch.Tensor]] = None,
    temperature: float = 1.0,
) -> DataProto:
    """
    将输入数据转换为DataProto格式
    
    Args:
        prompts: 输入提示列表
        responses: 模型输出列表
        rewards: 对应的reward列表
        tokenizer: tokenizer
        ref_log_probs: 参考策略的log_probs（可选）
        temperature: 温度参数
    """
    # Tokenize prompts and responses
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    response_tokens = tokenizer(responses, return_tensors="pt", padding=True, truncation=True)
    
    # 合并input_ids
    input_ids = torch.cat([prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1)
    
    # 创建attention_mask
    prompt_mask = prompt_tokens["attention_mask"]
    response_mask = response_tokens["attention_mask"]
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    
    # 创建position_ids
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    # 计算old_log_probs（使用当前模型，因为这是第一次更新）
    # 在实际使用中，如果是从已有数据来的，可能需要保存old_log_probs
    # 这里假设需要重新计算
    
    # 将rewards转换为token级别的rewards（简单实现：只在最后一个token处给reward）
    response_length = response_tokens["input_ids"].shape[1]
    token_level_rewards = torch.zeros(batch_size, response_length)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # 在最后一个有效token处放置reward
    for i in range(batch_size):
        valid_length = response_mask[i].sum().item()
        if valid_length > 0:
            token_level_rewards[i, valid_length - 1] = rewards_tensor[i]
    
    # 计算advantages（使用简单的REINFORCE方法，或者使用GAE）
    # 这里使用简单的REINFORCE：advantage = reward - baseline
    baseline = rewards_tensor.mean()
    advantages = (rewards_tensor - baseline).unsqueeze(-1).expand(-1, response_length)
    
    # 应用response_mask
    response_mask_tensor = response_mask.float()
    advantages = advantages * response_mask_tensor
    
    # 创建TensorDict
    batch_data = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": response_tokens["input_ids"],
        "advantages": advantages,
        "old_log_probs": torch.zeros(batch_size, response_length),  # 占位符，会在update_policy中重新计算
    }, batch_size=[batch_size])
    
    if ref_log_probs is not None:
        # 如果提供了ref_log_probs，使用它们
        ref_log_probs_tensor = torch.stack(ref_log_probs) if isinstance(ref_log_probs[0], torch.Tensor) else torch.tensor(ref_log_probs)
        batch_data["ref_log_probs"] = ref_log_probs_tensor
    
    # 创建DataProto
    dataproto = DataProto(
        batch=batch_data,
        non_tensor_batch={},
        meta_info={"temperature": temperature}
    )
    
    return dataproto


def compute_advantages_from_rewards(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    method: str = "reinforce",
    gamma: float = 1.0,
    lam: float = 0.95,
    values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    从rewards计算advantages
    
    Args:
        rewards: token级别的rewards，shape (batch_size, response_length)
        response_mask: response mask，shape (batch_size, response_length)
        method: 计算方法 ("reinforce", "gae", "grpo", "rloo")
        gamma: 折扣因子
        lam: GAE的lambda参数
        values: 价值函数预测（用于GAE）
    """
    if method == "reinforce":
        # 简单REINFORCE：advantage = reward - baseline
        # 将token级别的rewards求和得到每个样本的总reward
        total_rewards = (rewards * response_mask).sum(dim=-1)
        baseline = total_rewards.mean()
        advantages = (total_rewards - baseline).unsqueeze(-1).expand(-1, rewards.shape[1])
        advantages = advantages * response_mask
        return advantages
    
    elif method == "gae":
        # 使用GAE（需要values）
        if values is None:
            raise ValueError("GAE method requires values")
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=response_mask,
            gamma=torch.tensor(gamma),
            lam=torch.tensor(lam),
        )
        return advantages
    
    else:
        raise ValueError(f"Unknown method: {method}")


def direct_update_with_rewards(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
    actor_config: ActorConfig,
    optimizer: torch.optim.Optimizer,
    ref_log_probs: Optional[List[torch.Tensor]] = None,
    advantage_method: str = "reinforce",
    num_epochs: int = 1,
) -> Dict:
    """
    直接使用已有的输入输出和reward更新参数
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompts: 输入提示列表
        responses: 模型输出列表
        rewards: 对应的reward列表
        actor_config: Actor配置
        optimizer: 优化器
        ref_log_probs: 参考策略的log_probs（可选，用于KL散度）
        advantage_method: advantage计算方法
        num_epochs: 训练轮数
    
    Returns:
        训练指标字典
    """
    # 创建Actor
    actor = DataParallelPPOActor(
        config=actor_config,
        actor_module=model,
        actor_optimizer=optimizer,
    )
    
    # 准备数据
    dataproto = prepare_dataproto(
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        tokenizer=tokenizer,
        ref_log_probs=ref_log_probs,
        temperature=actor_config.temperature,
    )
    
    # 计算old_log_probs（使用当前模型）
    with torch.no_grad():
        old_log_probs = actor.compute_log_prob(dataproto)
        dataproto.batch["old_log_probs"] = old_log_probs
    
    # 如果需要，重新计算advantages
    if advantage_method != "reinforce":
        # 这里需要根据实际情况调整
        # 如果使用GAE，需要先计算values
        pass
    
    # 更新策略
    metrics = actor.update_policy(dataproto)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="直接使用已有的输入输出和reward更新参数")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="包含输入输出和reward的JSON文件")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--advantage_method", type=str, default="reinforce", 
                       choices=["reinforce", "gae"], help="advantage计算方法")
    
    args = parser.parse_args()
    
    # 加载模型和tokenizer
    print(f"加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    # 加载数据
    print(f"加载数据: {args.input_file}")
    data = load_data_from_file(args.input_file)
    
    prompts = [item["prompt"] for item in data]
    responses = [item["response"] for item in data]
    rewards = [item["reward"] for item in data]
    
    print(f"数据量: {len(prompts)}")
    print(f"平均reward: {sum(rewards) / len(rewards):.4f}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 创建Actor配置
    actor_config = ActorConfig(
        global_batch_size_per_device=args.batch_size,
        micro_batch_size_per_device_for_update=args.batch_size,
        micro_batch_size_per_device_for_experience=args.batch_size,
        ppo_epochs=args.num_epochs,
        temperature=1.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        use_kl_loss=True,
        kl_coef=0.1,
    )
    
    # 执行更新
    print("开始更新参数...")
    metrics = direct_update_with_rewards(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        actor_config=actor_config,
        optimizer=optimizer,
        advantage_method=args.advantage_method,
        num_epochs=args.num_epochs,
    )
    
    print("\n训练指标:")
    for key, value in metrics.items():
        if isinstance(value, list):
            print(f"  {key}: {sum(value) / len(value):.6f}")
        else:
            print(f"  {key}: {value:.6f}")
    
    # 保存模型
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n保存模型到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("完成！")


if __name__ == "__main__":
    main()

