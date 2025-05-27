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
        # MODIFICATION: Added bad_mask_space (can be simple Box(0,1,(1,)))
        # bad_mask_space: gym.Space # Example, if you want to define its space
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
        if args.use_centralized_V:
            share_agent_id_shape = get_shape_from_obs_space(share_agent_id_space)
        else:
            share_agent_id_shape = get_shape_from_obs_space(agent_id_space)
        adj_shape = get_shape_from_obs_space(adj_space)
        # MODIFICATION: Define bad_mask_shape, assuming it's (1,)
        bad_mask_shape = (1,)


        if type(obs_shape[-1]) == list: obs_shape = obs_shape[:1]
        if type(share_obs_shape[-1]) == list: share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.node_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *node_obs_shape), dtype=np.float32)
        self.adj = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *adj_shape), dtype=np.float32)
        self.agent_id = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *agent_id_shape), dtype=int)
        self.share_agent_id = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_agent_id_shape), dtype=int)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)

        act_shape = get_shape_from_act_space(act_space)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        # dones_env[step+1] stores logical done_t from environment
        self.dones_env = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        
        # MODIFICATION: Added bad_masks array
        # bad_masks[step+1] stores bad_mask_t (True if not a 'true' terminal state for value function)
        # This will be derived from info['bad_mask_indicator'] or info['is_absorb_state']
        self.bad_masks = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *bad_mask_shape), dtype=np.float32)


        self.step = 0

    # MODIFICATION: `insert` now accepts `bad_masks_from_info`
    def insert(
        self,
        share_obs: arr, obs: arr, node_obs: arr, adj: arr, agent_id: arr, share_agent_id: arr,
        actions: arr, action_log_probs: arr, value_preds: arr,
        rewards: arr,
        dones_env_logical: arr,   # Logical dones from env (True/False)
        bad_masks_from_info: arr, # Bad masks from info (True/False)
    ) -> None:
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.node_obs[self.step + 1] = node_obs.copy()
        self.adj[self.step + 1] = adj.copy()
        self.agent_id[self.step + 1] = agent_id.copy()
        self.share_agent_id[self.step + 1] = share_agent_id.copy()

        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        
        dones_float = dones_env_logical.astype(np.float32)
        if dones_float.ndim == 2:
            dones_float = np.expand_dims(dones_float, axis=-1)
        self.dones_env[self.step + 1] = dones_float.copy()

        # MODIFICATION: Store bad_masks
        # bad_masks_from_info is expected to be (n_threads, n_agents) bool
        # Convert to float and expand dims
        bad_masks_float = bad_masks_from_info.astype(np.float32)
        if bad_masks_float.ndim == 2:
            bad_masks_float = np.expand_dims(bad_masks_float, axis=-1)
        self.bad_masks[self.step + 1] = bad_masks_float.copy()
        
        self.step = (self.step + 1) % self.episode_length

    def after_update(self) -> None:
        self.share_obs[0] = self.share_obs[self.episode_length].copy()
        self.obs[0] = self.obs[self.episode_length].copy()
        self.node_obs[0] = self.node_obs[self.episode_length].copy()
        self.adj[0] = self.adj[self.episode_length].copy()
        self.agent_id[0] = self.agent_id[self.episode_length].copy()
        self.share_agent_id[0] = self.share_agent_id[self.episode_length].copy()
        
        # For the new s_0, it's not a 'done' start, and it's not a 'bad terminal' for GAE purposes initially
        self.dones_env[0] = np.zeros_like(self.dones_env[0])
        self.bad_masks[0] = np.ones_like(self.bad_masks[0]) # s_0 is generally a state where value should be bootstrapped

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:
        """
        计算回报 (GAE)
        next_value: V(s_L), a value estimate for the state after the last action in the segment.
                    Shape: (n_rollout_threads, num_agents, 1)
        """
        # V_old(s_L) is stored as self.value_preds[episode_length]
        # This is the value of the state obs[episode_length] which is s_L (successor of a_{L-1})
        self.value_preds[self.episode_length] = next_value.copy() 
        
        gae = 0.0
        for step_idx in reversed(range(self.episode_length)): # step_idx from L-1 down to 0
            # V_old(s_t)
            v_s_t_old = self.value_preds[step_idx]
            # V_old(s_{t+1}) or V_bootstrap(s_L) if t+1 == L
            v_s_t_plus_1_old = self.value_preds[step_idx + 1]
            
            # dones_env[step_idx + 1] is the logical done_t signal (0.0 or 1.0)
            # bad_masks[step_idx + 1] is the bad_mask_t signal (0.0 if true terminal, 1.0 if pseudo terminal for GAE)
            
            # Effective mask for GAE: if it's a bad_mask (pseudo terminal), we bootstrap.
            # If it's a logical done AND NOT a bad_mask (true terminal), we don't bootstrap.
            # In our case, info['bad_mask_indicator'] is always True from marl_env.
            # So, self.bad_masks will be all 1s if we directly use bad_mask_indicator.
            # This means we always bootstrap if the episode_length hasn't ended.
            # The crucial part is the mask applied to GAE's recursive term.
            # For GAE: delta_t = r_t + gamma * V(s_{t+1}) * (1-done_t_logical_if_terminal_else_0) - V(s_t)
            # gae_t = delta_t + gamma * lambda * gae_{t+1} * (1-done_t_logical_if_terminal_else_0)
            #
            # Let's use self.bad_masks[step_idx + 1] as the mask for bootstrapping V(s_{t+1})
            # And (1 - self.dones_env[step_idx + 1]) * self.bad_masks[step_idx + 1] as the mask for the gae recursive term.
            # Or more simply: if bad_mask is 1, it means we always bootstrap the value of V(s_{t+1}),
            #                  regardless of logical done.
            #                  The GAE recursion (gamma * lambda * gae * mask) also uses this bad_mask.

            bootstrap_mask_t1 = self.bad_masks[step_idx + 1] # Should be 1.0 if not a true absorbing state where V=0

            if self._use_popart or self._use_valuenorm:
                # delta = r_t + gamma * V_denorm(s_{t+1}) * bootstrap_mask_t1 - V_denorm(s_t)
                delta = (
                    self.rewards[step_idx]
                    + self.gamma * value_normalizer.denormalize(v_s_t_plus_1_old) * bootstrap_mask_t1
                    - value_normalizer.denormalize(v_s_t_old)
                )
                # gae = delta + gamma * lambda * bootstrap_mask_t1 * gae
                # The mask for gae recursion should also depend on whether it's a true terminal state.
                # If bad_mask is 1, it means the 'done' is not a true end, so gae recursion continues.
                gae = delta + self.gamma * self.gae_lambda * bootstrap_mask_t1 * gae
                self.returns[step_idx] = gae + value_normalizer.denormalize(v_s_t_old)
            else:
                delta = (
                    self.rewards[step_idx]
                    + self.gamma * v_s_t_plus_1_old * bootstrap_mask_t1
                    - v_s_t_old
                )
                gae = delta + self.gamma * self.gae_lambda * bootstrap_mask_t1 * gae
                self.returns[step_idx] = gae + v_s_t_old

    def feed_forward_generator( 
        self,
        advantages: arr,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator[
        Tuple[arr, arr, arr, arr, arr, arr, arr, arr, arr, arr, arr], # Added bad_masks_batch later if needed
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
        
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)
        
        # MODIFICATION: Also prepare bad_masks for batching if needed by the trainer
        # bad_masks = self.bad_masks[:-1].reshape(-1, *self.bad_masks.shape[3:]) # For s_0 to s_{L-1}'s transitions

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            node_obs_batch = node_obs[indices]
            adj_batch = adj[indices]
            agent_id_batch = agent_id[indices]
            share_agent_id_batch = share_agent_id[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = advantages[indices]
            # bad_masks_sample_batch = bad_masks[indices] # If needed

            yield (share_obs_batch, obs_batch, node_obs_batch, adj_batch,
                agent_id_batch, share_agent_id_batch, actions_batch,
                value_preds_batch, return_batch, old_action_log_probs_batch,
                adv_targ) # Add bad_masks_sample_batch here if trainer uses it