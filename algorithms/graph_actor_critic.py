import argparse
from typing import Any, Dict, List, Tuple

import gymnasium as gym 
import torch
import torch.nn as nn
from torch import Tensor
from algorithms.utils.util import check # TODO
from algorithms.utils.util import get_shape_frome_obs_space # TODO
from algorithms.utils.mlp import MLPBase # TODO
from algorithms.utils.gnn import GNNBase # TODO
from algorithms.utils.act import ACTLayer # TODO
from algorithms.utils.popart import PopArt # TODO 





class GR_Actor(nn.Module):
    def __init__(self,
                args: argparse.Namespace, # TODO
                 obs_space: gym.Space,
                 node_obs_space: gym.Space,
                 edge_obs_space: gym.Space,
                 action_space: gym.Space,
                 device = torch.device("cpu"),
                 )-> None:
        super(GR_Actor, self).__init__()

        self.args = args
        self.hidden_size = args.hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        self._use_orthogonal = args.use_orthogonal # 是否使用正交初始化
        self._gain = args.gain # 权重初始化的增益

        obs_shape = get_shape_frome_obs_space(obs_space) # TODO
        node_obs_shape = get_shape_frome_obs_space(node_obs_space)[1] # TODO
        edge_dim = get_shape_frome_obs_space(edge_obs_space)[0] # TODO
        gnn_out_dim = self.gnn_base.get_out_dim() # TODO

        self.gnn_base = GNNBase(
            args,
            node_obs_shape,
            edge_dim,
            args.actor_graph_aggr
        ) # TODO
  
        self.base = MLPBase(
            args,
            node_obs_shape,
            edge_dim,
            args.actor_graph_aggr,
        ) # TODO

        self.act = ACTLayer(
            action_space,
            self.hidden_size, 
            self._use_orthogonal, # 是否使用正交初始化 
            self._gain # 权重初始化的增益 
        ) # TODO

        self.to(device)

    def forward(
            self,
            obs: Dict[str, Tensor],
            node_obs: Dict[str, Tensor],
            adj: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            obs: observation dictionary
            node_obs: node observation dictionary
            adj: adjacency matrix

        """
        obs = check(obs).to(**self.tpdv) # TODO
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)

        nbd_features = self.gnn_base(node_obs, adj, agent_id)
        actor_features = torch.cat([obs, nbd_features], dim=1)
        actor_features = self.base(actor_features)

        actions, action_log_probs = self.act(actor_features) # TODO

        return (actions, action_log_probs)
    

    def evaluate_actions(
            self,
            obs: Dict[str, Tensor],
            node_obs: Dict[str, Tensor],
            adj: Tensor,
            agent_id: Tensor,
            actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
          

          obs = check(obs).to(**self.tpdv)
          node_obs = check(node_obs).to(**self.tpdv)
          adj = check(adj).to(**self.tpdv)
          agent_id = check(agent_id).to(**self.tpdv)
          actions = check(actions).to(**self.tpdv)

          nbd_features = self.gnn_base(node_obs, adj, agent_id)
          actor_features = torch.cat([obs, nbd_features], dim=1)
          actor_features = self.base(actor_features)

          action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, 
                actions,
                ) # TODO
          

          return (action_log_probs, dist_entropy)
    


class GR_Critic(nn.Module):






            return values