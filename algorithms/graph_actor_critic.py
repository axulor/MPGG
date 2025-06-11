import argparse
from typing import Tuple

import gym
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.gnn import GNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space


def minibatchGenerator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]  # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (edge_dim,)

        # GNN编码节点特征和边特征为向量
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)

        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]

        # 将GNN输出送入隐藏层
        self.base = MLPBase(args, input_dim = mlp_base_in_dim)

        # 把 MLP 隐藏层输出映射成动作分布参数
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, node_obs, adj, agent_id) :

        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(node_obs_batch, adj_batch, agent_id_batch)
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1) # 拼接GNN输出与自身观测
            actor_features = self.base(actor_features)


        actions, action_log_probs = self.act(actor_features)

        return actions, action_log_probs

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        action,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        node_obs (torch.Tensor):
            Local agent graph node features to the actor.
        adj (torch.Tensor):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)


        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action)

        return action_log_probs, dist_entropy


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
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal

        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        
        mlp_base_in_dim = gnn_out_dim
        
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, input_dim = mlp_base_in_dim)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, node_obs, adj, agent_id):
        """
        前向计算 Critic 的价值预测，兼容 node/global 两种聚合模式，并原地处理拆分小批次：
        - node 模式下 GNNBase 返回 [B,N,d]，按 agent_id 挑出 [B,d]
        - global 模式下 GNNBase 返回 [B,d]
        - （可选）拼接集中式观测 [B,C] → [B,C+d]
        - 过 MLPBase → 过输出层 → [B,1]
        - 若 split_batch=True 且 B > max_batch_size，会拆成多个子批次分别处理。
        """
        # 1. 转 tensor 并移动到 device
        cent_obs = check(cent_obs).to(**self.tpdv)   # [B, C]
        node_obs = check(node_obs).to(**self.tpdv)   # [B, N, D_node]
        adj      = check(adj).to(**self.tpdv)        # [B, N, N]
        agent_id = check(agent_id).to(**self.tpdv).long()  # [B] 或 [B,1]

        B = cent_obs.shape[0]

        # 2. 如果要拆小批次
        if self.split_batch and B > self.max_batch_size:
            critic_feats_list = []
            # minibatchGenerator 会返回若干 (obs_b, node_obs_b, adj_b, aid_b) 四元组
            for obs_b, node_obs_b, adj_b, aid_b in minibatchGenerator(
                    cent_obs, node_obs, adj, agent_id, self.max_batch_size):
                # 2.1 GNNBase 编码
                feats = self.gnn_base(node_obs_b, adj_b, aid_b)
                # 2.2 node 模式下：feats 可能是 [b,N,d]，索引取出 [b,d]
                if feats.dim() == 3:
                    b, N, d = feats.shape
                    idx = torch.arange(b, device=feats.device)
                    # aid_b 可能是 [b,1] 或 [b]
                    aid = agent_id[:, 0]
                    feats = feats[idx, aid, :]  # 取出对应行 => [b,d]
                # 2.3 拼接集中式观测（若启用）
                if self.args.use_cent_obs:
                    inp = torch.cat([obs_b, feats], dim=1)  # [b, C+d]
                else:
                    inp = feats                             # [b, d]
                # 2.4 过 MLPBase => [b, hidden_size]
                critic_feats_list.append(self.base(inp))
            # 2.5 拼回完整输出 => [B, hidden_size]
            critic_features = torch.cat(critic_feats_list, dim=0)
        else:
            # 3.1 一次性前向
            feats = self.gnn_base(node_obs, adj, agent_id)  # global->[B,d] or node->[B,N,d]
            # print(f"[DEBUG] GNNBase('{self.args.critic_graph_aggr}') 输出形状 = {feats.shape}")
            # 3.2 node 模式下挑行
            if feats.dim() == 3:
                b, N, d = feats.shape
                idx = torch.arange(b, device=feats.device)
                aid = agent_id[:, 0]
                feats = feats[idx, aid, :]  # => [B, d]
                # print(f"[DEBUG] GNNBase('{self.args.critic_graph_aggr}') 输出形状 = {feats.shape}")
            # 3.3 拼接集中式观测（若启用）
            if self.args.use_cent_obs:
                inp = torch.cat([cent_obs, feats], dim=1)  # [B, C+d]
            else:
                inp = feats                               # [B, d]
            # 3.4 过 MLPBase => [B, hidden_size]
            critic_features = self.base(inp)

        # 4. 最后过输出层 => [B,1]
        values = self.v_out(critic_features)
        return values



