# gnn_rl_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 RLlib 导入 TorchModelV2 基类
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2

# 引入自定义的图编码器与数据转换函数
from models.gnn_encoder import GNNEncoder

class GNNRLModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        初始化自定义模型
        
        参数:
            obs_space: 环境的观测空间，注意这里期望已经经过 graph_utils 转换为 PyG Data 对象格式
            action_space: 环境的动作空间
            num_outputs: 动作输出维度（即 actor_head 输出的 logits 维数）
            model_config: 模型配置参数
            name: 模型名称
        """
        # 初始化父类
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # 这里我们约定输入节点特征维度为 5（与 gnn_encoder.py 中的设计一致）
        in_channels = 5
        # 隐藏层维数（可根据需要调整）
        hidden_channels = 16
        # GNN 编码器最终输出的向量维数（例如 64 维）
        out_channels = 64
        
        # 创建 actor 和 critic 编码器，参数互不共享
        self.actor_encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.critic_encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        
        # actor_head：将编码器输出的特征转换为动作 logits
        self.actor_head = nn.Linear(out_channels, num_outputs)
        # critic_head：将编码器输出的特征转换为状态价值
        self.critic_head = nn.Linear(out_channels, 1)
        
        # 用于存储 forward() 时缓存的输入图数据（以便 value_function 调用）
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        """
        前向传播：计算 actor 部分的输出 logits
        
        参数:
            input_dict: 包含 "obs" 键，其值为经过 graph_utils 处理后的 PyG Data 对象
            state: 状态信息（如果有 RNN 等需求，这里可传递内部状态，本例为空）
            seq_lens: 序列长度（本例不使用）
        
        返回:
            logits: 动作选择的 logits，确保形状为 [batch, num_outputs]
            state: 原样返回 state
        """
        # 从输入字典中获取图数据
        graph_data = input_dict["obs"]
        # 缓存图数据，用于后续 value_function 调用
        self._last_obs = graph_data
        
        # 利用 actor_encoder 提取特征
        actor_features = self.actor_encoder(graph_data)
        # 通过 actor_head 得到动作 logits
        logits = self.actor_head(actor_features)
        # 若 logits 为 1 维，则增加 batch 维度，保证输出符合 RLlib 要求 [B, num_outputs]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits, state

    def value_function(self):
        """
        计算状态价值：通过 critic_encoder 和 critic_head 得到当前状态的价值估计
        
        注意:
            本方法利用 forward() 中缓存的图数据 self._last_obs。
            如果没有缓存数据，则返回一个零张量。
        
        返回:
            形状为 [B] 的状态价值张量
        """
        if self._last_obs is None:
            # 如果没有前向传播缓存，则返回 0
            return torch.tensor(0.0)
        # 利用 critic_encoder 提取特征
        critic_features = self.critic_encoder(self._last_obs)
        # 经由 critic_head 得到状态价值
        value = self.critic_head(critic_features)
        # 若输出为二维，需要压缩成 [B] 的形状
        if value.dim() == 2 and value.shape[0] == 1:
            value = value.squeeze(0)
        else:
            value = value.view(-1)
        return value
