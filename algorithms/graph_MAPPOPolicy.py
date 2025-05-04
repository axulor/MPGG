import gym
import argparse

import torch
from torch import Tensor
from typing import Tuple
from algorithms.graph_actor_critic import GR_Actor, GR_Critic
from utils.util import update_linear_schedule


class GR_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks
    to compute actions and value function predictions.

    args: (argparse.Namespace)
        Arguments containing relevant model and policy information.
    obs_space: (gym.Space)
        Observation space.
    cent_obs_space: (gym.Space)
        Value function input space
        (centralized input for MAPPO, decentralized for IPPO).
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space) a
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        act_space: gym.Space,
        device=torch.device("cpu"),
    ) -> None:
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.node_obs_space = node_obs_space
        self.edge_obs_space = edge_obs_space
        self.act_space = act_space
        self.split_batch = args.split_batch
        self.max_batch_size = args.max_batch_size

        self.actor = GR_Actor(
            args,
            self.obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.act_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )
        self.critic = GR_Critic(
            args,
            self.share_obs_space,
            self.node_obs_space,
            self.edge_obs_space,
            self.device,
            self.split_batch,
            self.max_batch_size,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        Decay the actor and critic learning rates.
        episode: (int)
            Current training episode.
        episodes: (int)
            Total number of training episodes.
        """
        update_linear_schedule(
            optimizer=self.actor_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.lr,
        )
        update_linear_schedule(
            optimizer=self.critic_optimizer,
            epoch=episode,
            total_num_epochs=episodes,
            initial_lr=self.critic_lr,
        )

    def get_actions(self, cent_obs, obs, node_obs, adj, agent_id, share_agent_id):

        actions, action_log_probs = self.actor.forward(obs, node_obs, adj, agent_id)

        values = self.critic.forward(cent_obs, node_obs, adj, share_agent_id)

        return values, actions, action_log_probs

    
    def get_values(self, cent_obs, node_obs, adj, share_agent_id):
        
        values = self.critic.forward(cent_obs, node_obs, adj, share_agent_id)

        return values

    def evaluate_actions(self, cent_obs, obs, node_obs,adj, agent_id, share_agent_id, action):

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs,
            node_obs,
            adj,
            agent_id,
            action)

        values = self.critic.forward(cent_obs, node_obs, adj, share_agent_id)
        return values, action_log_probs, dist_entropy

    def act(self,obs,node_obs,adj,agent_id) :

        actions, _ = self.actor.forward(obs, node_obs, adj, agent_id)
        
        return actions
