import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import gymnasium as gym
from algorithms.utils.mlp import MLPBase
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt # PopArt 仍然在输出层使用
from algorithms.utils.util import init, check # init 用于初始化 PopArt/Linear 输出层


class GR_Actor(nn.Module):
    """
    (修改版) Actor 网络类。接收个体观测和预处理的 GNN 特征，输出动作。
    不再内部管理 GNN。
    """
    def __init__(
        self,
        args: argparse.Namespace,
        actor_mlp_input_dim: int, # 个体观测维度 + GNN节点嵌入维度
        action_space: gym.Space,  # 动作空间用于 ACTLayer
        device=torch.device("cpu"),
        split_batch: bool = False, 
        max_batch_size: int = 1024, # 默认最大batch_size
    ) -> None:
        
        super(GR_Actor, self).__init__()

        self.args = args # 存储 args 以便 MLPBase 和 ACTLayer 使用
        self.hidden_size = args.hidden_size # MLPBase 和 ACTLayer 可能用到

        # 从 args 获取初始化相关的参数
        self.gain = args.gain
        self.use_orthogonal = args.use_orthogonal
        
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)


        # 实例化 MLPBase
        self.base = MLPBase(args, input_dim=actor_mlp_input_dim) # input_dim 是拼接后的总维度

        # 实例化 ACTLayer 
        self.act = ACTLayer(action_space, self.hidden_size, self.use_orthogonal, self.gain)

        self.to(device)


    def forward(self, 
                obs: Tensor,              # 个体观测, 形状 (B, D_obs)
                actor_gnn_feat: Tensor    # GNN处理后的节点嵌入, 形状 (B, D_actor_gnn_out)
            ) -> Tuple[Tensor, Tensor]:
        """
        接收个体观测和预计算的GNN节点嵌入, 拼接后通过MLP和ACTLayer输出动作。
        B 是当前处理的批次大小 (可能是 M*N, 或更小的 chunk)。
        """
        obs = check(obs).to(**self.tpdv)
        actor_gnn_feat = check(actor_gnn_feat).to(**self.tpdv)

        # 拼接个体观测和 GNN 输出的邻域特征
        combined_features = torch.cat([obs, actor_gnn_feat], dim=1)  # 形状 (B, D_obs + D_actor_gnn_out)

        # 对于较大的特征批次手动分块处理 MLP 部分
        if self.split_batch and combined_features.shape[0] > self.max_batch_size:
            actor_mlp_outputs_list = []
            num_chunks = combined_features.shape[0] // self.max_batch_size + \
                        (1 if combined_features.shape[0] % self.max_batch_size > 0 else 0)

            for i in range(num_chunks):
                chunk_start = i * self.max_batch_size
                chunk_end = min((i + 1) * self.max_batch_size, combined_features.shape[0])
                
                feature_chunk = combined_features[chunk_start:chunk_end]
                if feature_chunk.shape[0] == 0: continue

                mlp_output_chunk = self.base(feature_chunk) # 通过 MLPBase
                actor_mlp_outputs_list.append(mlp_output_chunk)
            
            mlp_output = torch.cat(actor_mlp_outputs_list, dim=0) # (B, self.hidden_size)
        else:
            # 直接通过 MLPBase
            mlp_output = self.base(combined_features) # (B, self.hidden_size)

        # 送入 ACTLayer 前向传播以获取动作和对数概率
        actions, action_log_probs = self.act(mlp_output)

        return actions, action_log_probs

    def evaluate_actions(
        self,
        obs: Tensor,              # 个体观测, 形状 (B, D_obs)
        actor_gnn_feat: Tensor,   # GNN处理后的节点嵌入, 形状 (B, D_actor_gnn_out)
        actions: Tensor,            # 实际执行的动作, 形状 (B, action_dim)
    ) -> Tuple[Tensor, Tensor]:
        """
        评估给定动作的对数概率和策略熵。
        B 是当前处理的批次大小。
        """
        obs = check(obs).to(**self.tpdv)
        actor_gnn_feat = check(actor_gnn_feat).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        # 拼接个体观测和 GNN 特征 
        combined_features = torch.cat([obs, actor_gnn_feat], dim=1)

        # 分块处理 MLP (与 forward 方法中相同)
        if self.split_batch and combined_features.shape[0] > self.max_batch_size:
            log_probs_list = []
            dist_entropy_list = [] # 熵通常是标量或每个样本一个值
            
            num_chunks = combined_features.shape[0] // self.max_batch_size + \
                        (1 if combined_features.shape[0] % self.max_batch_size > 0 else 0)

            for i in range(num_chunks):
                chunk_start = i * self.max_batch_size
                chunk_end = min((i + 1) * self.max_batch_size, combined_features.shape[0])

                feature_chunk = combined_features[chunk_start:chunk_end]
                action_chunk = actions[chunk_start:chunk_end] # 同步拆分动作
                
                if feature_chunk.shape[0] == 0: continue

                mlp_output_chunk = self.base(feature_chunk)
                log_probs_chunk, dist_entropy_chunk = self.act.evaluate_actions(mlp_output_chunk, action_chunk)
                
                log_probs_list.append(log_probs_chunk)
                dist_entropy_list.append(dist_entropy_chunk) # dist_entropy 通常是 (B_chunk,) 或标量
            
            action_log_probs = torch.cat(log_probs_list, dim=0)
            # 处理 dist_entropy: 如果它是每个样本一个值，则cat；如果是标量，则取平均
            if dist_entropy_list[0].shape: # 如果不是标量
                dist_entropy = torch.cat(dist_entropy_list, dim=0).mean() # 或不取mean，取决于ACTLayer返回
            else: # 如果是标量列表
                dist_entropy = torch.tensor(dist_entropy_list).mean()

            dist_entropy_values = torch.stack(dist_entropy_list) # (num_chunks)
            dist_entropy = dist_entropy_values.mean() # 平均所有minibatch的熵均值


        else: #不拆分
            mlp_output = self.base(combined_features)
            action_log_probs, dist_entropy = self.act.evaluate_actions(mlp_output, actions)
            if dist_entropy.shape: # 取平均
                dist_entropy = dist_entropy.mean()


        return action_log_probs, dist_entropy


class GR_Critic(nn.Module):
    """
    Critic 网络类。接收中心化观测和预处理的图级全局GNN特征, 输出价值预测
    """
    def __init__(
        self,
        args: argparse.Namespace,
        critic_mlp_input_dim: int, # 中心化观测维度 + 全局GNN嵌入维度 
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 1024, 
    ) -> None:
        
        super(GR_Critic, self).__init__()

        self.args = args
        self.hidden_size = args.hidden_size # MLPBase 和 PopArt/Linear 输出层可能用到
        self.use_orthogonal = args.use_orthogonal # 用于输出层初始化

        self.use_popart = args.use_popart # PopArt 的使用与否
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # 初始化方法选择
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]


        # MLPBase 直接接收组合后的特征维度
        self.base = MLPBase(args, input_dim=critic_mlp_input_dim)

        # 输出层 (v_out)
        def init_linear_or_popart(m_layer): # 辅助函数简化初始化
            return init(m_layer, init_method, lambda x: nn.init.constant_(x, 0))

        if self.use_popart:
            self.v_out = init_linear_or_popart(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_linear_or_popart(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, 
                share_obs: Tensor,        # 中心化/个体观测, 形状 (B, D_share_obs)
                critic_gnn_feat: Tensor     # GNN处理后的图级全局嵌入, 形状 (B, D_critic_gnn_global_out)
                ) -> Tensor:
        """
        接收中心化观测和预计算的图级全局GNN嵌入。
        根据 args.use_cent_obs 决定是否拼接它们, 然后通过MLP和输出层得到价值预测。
        B 是当前处理的批次大小。
        """
        share_obs = check(share_obs).to(**self.tpdv)
        critic_gnn_feat = check(critic_gnn_feat).to(**self.tpdv)

        # 配置组合特征
        if self.args.use_cent_obs:  # 拼接全局观测信息和全局图特征
            combined_features = torch.cat([share_obs, critic_gnn_feat], dim=1)
        else: # 如果不使用额外的中心化观测
            combined_features = critic_gnn_feat

        # 手动分块处理 MLP 部分 (与 GR_Actor.forward 中类似)
        if self.split_batch and combined_features.shape[0] > self.max_batch_size:
            critic_mlp_outputs_list = []
            num_chunks = (combined_features.shape[0] + self.max_batch_size - 1) // self.max_batch_size

            for i in range(num_chunks):
                chunk_start = i * self.max_batch_size
                chunk_end = min((i + 1) * self.max_batch_size, combined_features.shape[0])
                
                feature_chunk = combined_features[chunk_start:chunk_end]
                if feature_chunk.shape[0] == 0: continue

                mlp_output_chunk = self.base(feature_chunk) # 通过 MLPBase
                critic_mlp_outputs_list.append(mlp_output_chunk)
            
            final_mlp_output = torch.cat(critic_mlp_outputs_list, dim=0) # (B, self.hidden_size)
        else:
            # 直接通过 MLPBase
            final_mlp_output = self.base(combined_features) # (B, self.hidden_size)

        # 通过价值输出层得到价值预测
        values = self.v_out(final_mlp_output) # (B, 1)

        return values



