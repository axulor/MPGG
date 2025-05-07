import torch
import gym
import argparse
import numpy as np
from numpy import ndarray as arr
from typing import Optional, Tuple, Generator
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class GraphReplayBuffer(object):
    """
    Buffer to store training data. For graph-based environments
    args: (argparse.Namespace)
        arguments containing relevant model, policy, and env information.
    num_agents: (int)
        number of agents in the env.
    num_entities: (int)
        number of entities in the env. This will be used for the `edge_list`
        size and `node_feats`
    obs_space: (gym.Space)
        observation space of agents.
    cent_obs_space: (gym.Space)
        centralized observation space of agents.
    node_obs_space: (gym.Space)
        node observation space of agents.
    agent_id_space: (gym.Space)
        observation space of agent ids.
    share_agent_id_space: (gym.Space)
        centralised observation space of agent ids.
    adj_space: (gym.Space)
        observation space of adjacency matrix.
    act_space: (gym.Space)
        action space for agents.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        num_agents: int,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        agent_id_space: gym.Space,
        share_agent_id_space: gym.Space,
        adj_space: gym.Space,
        act_space: gym.Space,
    ):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        # get shapes of observations
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)
        agent_id_shape = get_shape_from_obs_space(agent_id_space)
        if args.use_centralized_V:
            share_agent_id_shape = get_shape_from_obs_space(share_agent_id_space)
        else:
            share_agent_id_shape = get_shape_from_obs_space(agent_id_space)
        adj_shape = get_shape_from_obs_space(adj_space)
        ####################

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32,
        )
        # graph related stuff
        self.node_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *node_obs_shape,
            ),
            dtype=np.float32,
        )
        self.adj = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *adj_shape),
            dtype=np.float32,
        )
        self.agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *agent_id_shape,
            ),
            dtype = int,
        )
        self.share_agent_id = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_agent_id_shape,
            ),
            dtype = int,
        )
        ####################


        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.returns = np.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs: arr,
        obs: arr,
        node_obs: arr,
        adj: arr,
        agent_id: arr,
        share_agent_id: arr,
        actions: arr,
        action_log_probs: arr,
        value_preds: arr,
        rewards: arr,
        bad_masks: arr = None,
        active_masks: arr = None,
        available_actions: arr = None,
    ) -> None:
        """
        Insert data into the buffer.
        share_obs: (argparse.Namespace)
            arguments containing relevant model, policy, and env information.
        obs: (np.ndarray)
            local agent observations. [num_rollouts, num_agents, obs_shape]
        node_obs: (np.ndarray)
            node features for the graph.
        adj: (np.ndarray)
            adjacency matrix for the graph.
            NOTE: needs post-processing to split
            into edge_feat and edge_attr
        agent_id: (np.ndarray)
            the agent id †o which the observation belong to
        share_agent_id: (np.ndarray)
            the agent id to which the shared_observations belong to
        rnn_states_actor: (np.ndarray)
            RNN states for actor network.
        rnn_states_critic: (np.ndarray)
            RNN states for critic network.
        actions:(np.ndarray)
            actions taken by agents.
        action_log_probs:(np.ndarray)
            log probs of actions taken by agents
        value_preds: (np.ndarray)
            value function prediction at each step.
        rewards: (np.ndarray)
            reward collected at each step.
        masks: (np.ndarray)
            denotes whether the environment has terminated or not.
        bad_masks: (np.ndarray)
            action space for agents.
        active_masks: (np.ndarray)
            denotes whether an agent is active or dead in the env.
        available_actions: (np.ndarray)
            actions available to each agent.
            If None, all actions are available.
        """
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
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self) -> None:
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.node_obs[0] = self.node_obs[-1].copy()
        self.adj[0] = self.adj[-1].copy()
        self.agent_id[0] = self.agent_id[-1].copy()
        self.share_agent_id[0] = self.share_agent_id[-1].copy()

    def compute_returns(
        self, next_value: arr, value_normalizer: Optional[PopArt] = None
    ) -> None:
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        next_value: (np.ndarray)
            value predictions for the step after the last episode step.
        value_normalizer: (PopArt)
            If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self,
        advantages: arr,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Generator[
        Tuple[arr,arr,arr,arr,arr,arr,arr,arr,arr,arr,arr],
        None,
        None,
    ]:
        """
        Yield training data for MLP policies.
        advantages: (np.ndarray)
            advantage estimates.
        num_mini_batch: (int)
            number of minibatches to split the batch into.
        mini_batch_size: (int)
            number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) * number of agents "
                f"({num_agents}) = {n_rollout_threads*episode_length*num_agents} "
                "to be greater than or equal to the number of "
                f"PPO mini batches ({num_mini_batch})."
            )
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
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
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
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, node_obs_batch, adj_batch, \
                    agent_id_batch, share_agent_id_batch,  actions_batch, value_preds_batch, \
                    return_batch, old_action_log_probs_batch, adv_targ

