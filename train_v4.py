import os
import ray
import numpy as np
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from gymnasium.spaces import Dict, Box

from envs.migratory_pgg_env_v4 import MigratoryPGGEnv
from models.gnn_model import GNNEncoder


# 注册自定义模型
ModelCatalog.register_custom_model("gnn_encoder", GNNEncoder)

# 注册环境
def env_creator(config):
    return MigratoryPGGEnv(**config)

register_env("migratory_pgg", env_creator)

# 环境参数
env_config = {
    "N": 100,
    "max_cycles": 500,
    "size": 20,
    "speed": 0.1,
    "radius": 2.0,
    "cost": 1.0,
    "r": 3.0,
    "beta": 1.0,
    "seed": 47
}


# 初始化 Ray
ray.init()

# 获取 obs/act space
dummy_env = MigratoryPGGEnv(**env_config)
obs_space = dummy_env.observation_space(dummy_env.agents[0])
act_space = dummy_env.action_space(dummy_env.agents[0])
print(">>> obs_space:", obs_space)

# 构建配置
# 指定 GNN 编码后的向量维度（和模型一致）
encoded_obs_space = Box(low=-1.0, high=1.0, shape=(64,), dtype=np.float32)

config = (
    PPOConfig()
    .environment(env="migratory_pgg", env_config=env_config)
    .framework("torch")
    .env_runners(num_env_runners=0)
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .multi_agent(
        policies={
            "shared_policy": PolicySpec(
                observation_space=encoded_obs_space,  # ❗️注意这里
                action_space=act_space,
                config={}
            )
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
)



# 模型配置
config.model = {
    "custom_model": "gnn_encoder",
    "custom_model_config": {
        "hidden_dim": 64,
    },
    "vf_share_layers": True,
}


# 强化学习超参数配置
config.gamma = 0.99
config.lr = 5e-4
config.train_batch_size = 4000
config.sgd_minibatch_size = 128
config.num_sgd_iter = 10


# 启动训练
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        name="migratory_pgg_training",
        stop={"training_iteration": 100},
        storage_path=Path("results").resolve().as_posix()
    )
)

results = tuner.fit()
ray.shutdown()
