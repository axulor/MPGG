import torch
import gym # 导入 gym，因为 obs_space 等类型提示中用到
import argparse
import numpy as np
from numpy import ndarray as arr
from typing import Optional, Tuple, Generator
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


class GraphReplayBuffer(object):
    """
    用于存储图环境训练数据的缓冲区。
    通过循环写入NumPy数组, 实现逻辑上的滑动窗口效果, 以支持连续的无限时域环境数据收集。
    """

    def __init__(
        self,
        args: argparse.Namespace,
        num_agents: int,
        obs_space: gym.Space,               # 智能体的观测空间
        cent_obs_space: gym.Space,          # 中心化观测空间
        node_obs_space: gym.Space,          # 节点观测空间
        agent_id_space: gym.Space,          # 智能体ID空间
        share_agent_id_space: gym.Space,    # 共享智能体ID空间
        adj_space: gym.Space,               # 邻接矩阵空间
        act_space: gym.Space,               # 动作空间
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)
        agent_id_shape = get_shape_from_obs_space(agent_id_space)
        share_agent_id_shape = get_shape_from_obs_space(share_agent_id_space)
        adj_shape = get_shape_from_obs_space(adj_space)



        if type(obs_shape[-1]) == list: obs_shape = obs_shape[:1]
        if type(share_obs_shape[-1]) == list: share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.node_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *node_obs_shape), dtype=np.float32)
        self.adj = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *adj_shape), dtype=np.float32)
        self.agent_id = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *agent_id_shape), dtype=int)
        self.share_agent_id = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_agent_id_shape), dtype=int)

        self.values = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.values)

        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.dones = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)


        self.step = 0

    def insert(
        self,
        share_obs: arr, obs: arr, node_obs: arr, adj: arr, agent_id: arr, share_agent_id: arr,
        actions: arr, action_log_probs: arr, values: arr,
        rewards: arr,
        dones: arr,
    ) -> None:
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.node_obs[self.step + 1] = node_obs.copy()
        self.adj[self.step + 1] = adj.copy()
        self.agent_id[self.step + 1] = agent_id.copy()
        self.share_agent_id[self.step + 1] = share_agent_id.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.values[self.step] = values.copy()
        self.rewards[self.step] = rewards.copy()

        dones = np.expand_dims(dones.astype(np.float32), axis=-1) # 转换为浮点数并扩展一维           
        self.dones[self.step + 1] = dones.copy()

        
        self.step = (self.step + 1) % self.episode_length # 循环写入指针+1

    def after_update(self) -> None:
        self.share_obs[0] = self.share_obs[self.episode_length].copy()
        self.obs[0] = self.obs[self.episode_length].copy()
        self.node_obs[0] = self.node_obs[self.episode_length].copy()
        self.adj[0] = self.adj[self.episode_length].copy()
        self.agent_id[0] = self.agent_id[self.episode_length].copy()
        self.share_agent_id[0] = self.share_agent_id[self.episode_length].copy()
        self.dones[0] = np.zeros_like(self.dones[0])

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:

        self.values[self.episode_length] = next_value.copy() # 存储末位的价值估计
        
        gae = 0.0
        for step_idx in reversed(range(self.episode_length)): # step_idx from L-1 down to 0
            v_s_t_old = self.values[step_idx]
            v_s_t_plus_1_old = self.values[step_idx + 1]

            if self._use_popart or self._use_valuenorm:
                # delta = r_t + gamma * V_denorm(s_{t+1})  - V_denorm(s_t)
                delta = (
                    self.rewards[step_idx]
                    + self.gamma * value_normalizer.denormalize(v_s_t_plus_1_old) 
                    - value_normalizer.denormalize(v_s_t_old)
                )
                gae = delta + self.gamma * self.gae_lambda  * gae
                self.returns[step_idx] = gae + value_normalizer.denormalize(v_s_t_old)
            else:
                delta = (
                    self.rewards[step_idx]
                    + self.gamma * v_s_t_plus_1_old 
                    - v_s_t_old
                )
                gae = delta + self.gamma * self.gae_lambda  * gae
                self.returns[step_idx] = gae + v_s_t_old

    def feed_forward_generator( 
        self,
        advantages: arr,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator[
        Tuple[arr, arr, arr, arr, arr, arr, arr, arr, arr, arr, arr], 
        None,
        None,
    ]:

        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, "PPO requires batch_size >= num_mini_batch"
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        node_obs = self.node_obs[:-1].reshape(-1, *self.node_obs.shape[3:])
        adj = self.adj[:-1].reshape(-1, *self.adj.shape[3:])
        agent_id = self.agent_id[:-1].reshape(-1, *self.agent_id.shape[3:])
        share_agent_id = self.share_agent_id[:-1].reshape(-1, *self.share_agent_id.shape[3:])
        
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        
        values = self.values[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            node_obs_batch = node_obs[indices]
            adj_batch = adj[indices]
            agent_id_batch = agent_id[indices]
            share_agent_id_batch = share_agent_id[indices]
            actions_batch = actions[indices]
            values_batch = values[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]

            yield (share_obs_batch, obs_batch, node_obs_batch, adj_batch,
                agent_id_batch, share_agent_id_batch, actions_batch,
                values_batch, return_batch, old_action_log_probs_batch,
                adv_targ) 