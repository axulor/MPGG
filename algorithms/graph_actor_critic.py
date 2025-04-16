import argparse
from typing import  Dict, Tuple

import gymnasium as gym 
import torch
import torch.nn as nn
from torch import Tensor
from algorithms.utils.util import init, check, get_shape_frome_obs_space, minibatchGenerator, init_  
from algorithms.utils.gnn import GNNBase 
from algorithms.utils.mlp import MLPBase 
from algorithms.utils.act import ACTLayer 




class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.  
    """
    def __init__(self,
                obs_space: gym.Space, # 观测空间
                action_space: gym.Space, # 动作空间
                device = torch.device("cpu"), 
                )-> None:
        
        super(GR_Actor, self).__init__()

        self.gnn_base = GNNBase() # TODO
        
        self.mlp_base = MLPBase() # TODO
        
        self.act_layer = ACTLayer() # TODO

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

        def forward(
            self,
            obs,
            node_obs,
            adj,
            agent_id
        ) -> Tuple[Tensor, Tensor, Tensor]:
            obs = check(obs).to(**self.tpdv)
            node_obs = check(node_obs).to(**self.tpdv)
            adj = check(adj).to(**self.tpdv)
            agent_id = check(agent_id).to(**self.tpdv).long()
            

    
            nbd_features = self.gnn_base(node_obs, adj, agent_id) #TODO 提取指定智能体的邻居特征
            actor_features = torch.cat([obs, nbd_features], dim=1) # 拼接自身+邻居特征
            actor_features = self.mlp_base(actor_features) # 特征送入神经网络
            actions, action_log_probs = self.act_layer(actor_features) # 输出动作和动作概率分布

            return (actions, action_log_probs)
    


class GR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    """
    def __init__(
            self,
            device = torch.device("cpu"),
        )-> None:

        super(GR_Critic, self).__init__()

        self.gnn_base = GNNBase() # TODO
        self.mlp_base = MLPBase() # TODO
        self.v_out = init_(nn.Linear(self.hidden_size, 1)) # TODO
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self):
        
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)

        nbd_features = self.gnn_base()
        
        critic_features = self.mlp_base(nbd_features)

        values = self.v_out(critic_features)

        return values # TODO
