# train/train_ippo.py

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from gymnasium.envs.registration import register

from envs.migratory_pgg_env_v4 import MigratoryPGGEnv
from models.gnn_rl_model import GNNRLModel

# 1. 注册环境到 Gym
register(
    id="MigratoryPGG-v0",
    entry_point="migratory_pgg_env_v4:MigratoryPGGEnv",
)

# 2. 注册自定义模型到 RLlib
ModelCatalog.register_custom_model("gnn_rl_model", GNNRLModel)

# 3. 创建环境实例，以便获取各 agent 的观测和动作空间
env_instance = MigratoryPGGEnv()

# 为每个 agent 创建独立策略
policies = {}
for agent in env_instance.possible_agents:
    policies[f"policy_{agent}"] = (
        None,  # 默认策略类
        env_instance.observation_spaces[agent],
        env_instance.action_spaces[agent],
        {}  # 额外配置
    )

def policy_mapping_fn(agent_id, episode, **kwargs):
    return f"policy_{agent_id}"

# 4. 构造 RLlib 配置字典（关闭新 API 堆栈）
config = {
    "env": "MigratoryPGG-v0",
    "num_workers": 1,
    "framework": "torch",
    "rollout_fragment_length": 100,  # 旧 API 方式指定采样片段长度
    "train_batch_size": 500,
    "lr": 5e-4,
    "model": {
        "custom_model": "gnn_rl_model",
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "api_stack": {
        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False
    },
    "experimental": {
        "_disable_rl_module_api": True,
        "_validate_config": False
    },
}

if __name__ == "__main__":
    ray.init()
    tune.run("PPO", config=config)
