# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import ray
import torch
import numpy as np
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        # 输出一条数据的原始数据和 message 数据，用于检查数据加载是否正确
        # print("\n" + "="*80)
        # print("检查数据加载：输出第一条样本的原始数据和构造的 messages")
        # print("="*80)
        
        # # 从 dataloader 获取第一条数据
        # sample = next(iter(train_dataloader))
        
        # # 获取原始数据（从 dataset 的底层数据集）
        # train_dataset = train_dataloader.dataset
        # if hasattr(train_dataset, 'dataset') and len(train_dataset.dataset) > 0:
        #     raw_example = train_dataset.dataset[0]
        #     print("\n【原始数据 (raw_example)】")
        #     print(json.dumps(raw_example, indent=2, ensure_ascii=False, default=str))
            
        #     # 构造 messages
        #     messages = train_dataset._build_messages(raw_example)
        #     print("\n【构造的 Messages】")
        #     print(json.dumps(messages, indent=2, ensure_ascii=False, default=str))
        
        # # 输出处理后的数据样本（包含 input_ids, ground_truth 等）
        # print("\n【处理后的样本数据 (sample)】")
        # print(f"Keys in sample: {list(sample.keys())}")
        
        # # 输出 ground_truth（对于 kernelbench 应该是 raw_problem）
        # if "ground_truth" in sample:
        #     ground_truth = sample["ground_truth"]
        #     # 处理 numpy array 或 list/tuple
        #     if isinstance(ground_truth, np.ndarray):
        #         ground_truth = ground_truth.tolist()
        #     if isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
        #         gt_val = ground_truth[0] if isinstance(ground_truth[0], str) else str(ground_truth[0])
        #         print(f"\n【ground_truth (第一条)】")
        #         print(f"Type: {type(ground_truth[0])}")
        #         print(f"Length: {len(gt_val)} characters" if isinstance(gt_val, str) else "N/A")
        #         print(f"Content (前500字符):\n{gt_val[:500]}..." if isinstance(gt_val, str) and len(gt_val) > 500 else gt_val)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()

      

      


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())
    with open('tokens.json', 'r') as f:
        tokens = json.load(f)
    os.environ['HF_TOKEN'] = tokens['huggingface']
    os.environ['WANDB_API_KEY'] = tokens['wandb']
    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        # 从环境变量中读取 CUDA_SERVER_IP（如果设置了）
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PYTHONUNBUFFERED": "1",
        }
        # 如果设置了 CUDA_SERVER_IP，传递给 Ray worker
        if "CUDA_SERVER_IP" in os.environ:
            env_vars["CUDA_SERVER_IP"] = os.environ["CUDA_SERVER_IP"]
        
        runtime_env = {
            "env_vars": env_vars
        }
        # 尝试连接到已存在的 Ray 集群，如果失败则创建新集群
        try:
            ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
        except (ConnectionError, RuntimeError) as e:
            print(f"Failed to connect to existing Ray cluster: {e}")
            print("Creating new Ray cluster...")
            
    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
