import torch
import gymnasium as gym 
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
        obs_space: gym.Space,               # 智能体的观测空间
        adj_space: gym.Space,               # 邻接矩阵空间
        act_space: gym.Space,               # 动作空间
    ):
        # 参数接收
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.use_popart = args.use_popart
        self.use_valuenorm = args.use_valuenorm
        self.num_agents =  args.num_agents
        
        # 从 Gym Space 获取原始形状
        full_obs_shape = (self.num_agents, *get_shape_from_obs_space(obs_space))
        adj_shape = get_shape_from_obs_space(adj_space)
        act_shape = get_shape_from_act_space(act_space)
        log_prob_shape = (1,)

        # 声明存储的参数
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *full_obs_shape), dtype=np.float32)
        self.adj = np.zeros((self.episode_length + 1, self.n_rollout_threads, *adj_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.values = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.values)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, *log_prob_shape), dtype=np.float32)
        self.dones = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.step = 0

    def insert(self, obs, adj, actions, action_log_probs, values, rewards, dones):

        self.obs[self.step + 1] = obs.copy()
        self.adj[self.step + 1] = adj.copy() # 同个环境，所有节点存储的距离邻接矩阵都相同
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.values[self.step] = values.copy()
        self.rewards[self.step] = rewards.copy()
        self.dones[self.step + 1] = np.expand_dims(dones.astype(np.float32), axis=-1)

        self.step = (self.step + 1) % self.episode_length # 循环写入指针+1

    def after_update(self) -> None:
        self.obs[0] = self.obs[self.episode_length].copy()
        self.adj[0] = self.adj[self.episode_length].copy()
        self.dones[0].fill(0.0)

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:

        self.values[self.episode_length] = next_value.copy() # 存储末位的价值估计
        
        gae = 0.0
        for step in reversed(range(self.episode_length)): # step_idx from L-1 down to 0

            if self.use_valuenorm and value_normalizer is not None:
                v_s_t_plus_1 = value_normalizer.denormalize(self.values[step + 1])
                v_s_t = value_normalizer.denormalize(self.values[step])
            else: # for PopArt or no normalization
                v_s_t_plus_1 = self.values[step + 1]
                v_s_t = self.values[step]

            delta = self.rewards[step] + self.gamma * v_s_t_plus_1 * (1 - self.dones[step + 1]) - v_s_t
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step + 1]) * gae
            self.returns[step] = gae + v_s_t

    def feed_forward_generator(self, advantages, mini_batch_size):

        episode_length, n_rollout_threads, num_agents, obs_dim = self.obs.shape
        episode_length -= 1

        batch_size = n_rollout_threads * episode_length * num_agents
        num_mini_batch = batch_size // mini_batch_size

        # 生成扁平化样本
        rand_indices = torch.randperm(batch_size).numpy() # 样本的随机索引

        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            minibatch_indices = rand_indices[start:end]

            t_indices = minibatch_indices // (n_rollout_threads * num_agents)
            m_indices = (minibatch_indices // num_agents) % n_rollout_threads
            n_indices = minibatch_indices % num_agents
            obs_batch = self.obs[t_indices, m_indices, n_indices]
            
            node_obs_batch = self.obs[t_indices, m_indices]
            adj_batch = self.adj[t_indices, m_indices] 
            # Get shared data for the minibatch
            agent_id_batch = n_indices.reshape(-1, 1)
            env_id_batch = m_indices.reshape(-1, 1)

            # Other agent-specific data
            actions_batch = self.actions[t_indices, m_indices, n_indices]
            action_log_probs_batch = self.action_log_probs[t_indices, m_indices, n_indices]
            values_batch = self.values[t_indices, m_indices, n_indices]
            returns_batch = self.returns[t_indices, m_indices, n_indices]
            # advantages are pre-flattened, so simple indexing is fine
            advantages_batch = advantages.reshape(-1, 1)[minibatch_indices]

            # print(f"[DEBUG]  obs_batch: {obs_batch.shape}, obs_batch: {obs_batch.dtype}")
            # print(f"[DEBUG]  node_obs_batch: {node_obs_batch.shape}, node_obs_batch: {node_obs_batch.dtype}")
            # print(f"[DEBUG]  adj_batch: {adj_batch.shape}, adj_batch: {adj_batch.dtype}")
            # print(f"[DEBUG]  agent_id_batch: {agent_id_batch.shape}, agent_id_batch: {agent_id_batch.dtype}")
            # print(f"[DEBUG]  env_id_batch: {env_id_batch.shape}, env_id_batch: {env_id_batch.dtype}")
            # print(f"[DEBUG]  actions_batch: {actions_batch.shape}, actions_batch: {actions_batch.dtype}")
            # print(f"[DEBUG]  values_batch: {values_batch.shape}, values_batch: {values_batch.dtype}")
            # print(f"[DEBUG]  returns_batch: {returns_batch.shape}, returns_batch: {returns_batch.dtype}")
            # print(f"[DEBUG]  action_log_probs_batch: {action_log_probs_batch.shape}, action_log_probs_batch: {action_log_probs_batch.dtype}")
            # print(f"[DEBUG]  advantages_batch: {advantages_batch.shape}, advantages_batch: {advantages_batch.dtype}")
            # print(f"\n")

            yield (obs_batch, node_obs_batch, adj_batch,
                agent_id_batch, env_id_batch, actions_batch,
                values_batch, returns_batch, action_log_probs_batch,
                advantages_batch) 