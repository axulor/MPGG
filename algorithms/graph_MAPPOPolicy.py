import gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from algorithms.graph_actor_critic import GR_Actor, GR_Critic
from algorithms.utils.util import update_linear_schedule

class GR_MAPPOPolicy:
    """
    GNN-based MAPPO Policy wrapper
    """

    def __init__(self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        share_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        act_space: gym.Space,
        device=torch.device("cpu"),
        ):

        # 解析超参数
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space 
        self.share_obs_space = share_obs_space 
        self.node_obs_space = node_obs_space
        self.edge_obs_space = edge_obs_space
        self.act_space = act_space

        self.device = device # 默认为 cpu
        self.split_batch = args.split_batch         # TODO    
        self.max_batch_size = args.max_batch_size   # TODO


        self.actor = GR_Actor(
            args,
            self.obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.act_space,
            self.device,
            self.split_batch,   # TODO
            self.max_batch_size,# TODO
        )
        self.critic = GR_Critic(
            args,
            self.share_obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.device,
            self.split_batch,   # TODO
            self.max_batch_size,# TODO
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """
        降低 actor 和 critic 的学习率
        """
        update_linear_schedule(
            optimizer = self.actor_optimizer,
            epoch = episode,
            total_num_epochs = episodes,
            initial_lr = self.actor_lr,
        )
        update_linear_schedule(
            optimizer = self.critic_optimizer,
            epoch = episode,
            total_num_epochs = episodes,
            initial_lr = self.critic_lr,
        )
        
    def get_values(self, 
        share_obs, 
        node_obs, 
        adj, 
        share_agent_id,
        ):
        """
        仅返回 critic 的值
        """
        values = self.critic.forward(
            share_obs, 
            node_obs, 
            adj, 
            share_agent_id, # TODO
        )
        return values

    def act(self,
        obs,
        node_obs,
        adj,
        agent_id,
        ):
        """
        返回 actor 给出的动作
        """
        actions, _ = self.actor.forward(
            obs,
            node_obs,
            adj,
            agent_id,
        )
        return actions
    
    def get_actions(self,
        obs,
        share_obs,
        node_obs,
        adj,
        agent_id,
        share_agent_id, # TODO
        ):
        """
        返回 critic 的值  +  actor 的动作
        """
        actions, action_log_probs = self.actor.forward(
            obs,
            node_obs,
            adj,
            agent_id,
        )

        values = self.critic.forward(
            share_obs, 
            node_obs, 
            adj, 
            share_agent_id, # TODO
        )
        return (values, actions, action_log_probs)
    
    def evaluate_actions(self,
        obs,
        share_obs,
        node_obs,
        adj,
        agent_id,
        share_agent_id, # TODO
        action,         # TODO
        ):
        """
        评估已采取动作的 log 概率、熵，计算 critic 值
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            node_obs,
            adj,
            agent_id,
            action,
        )

        values = self.critic.forward(
            share_obs, 
            node_obs, 
            adj, 
            share_agent_id,
        )

        return values, action_log_probs, dist_entropy

