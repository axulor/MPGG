import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import init, get_clones
import argparse
from typing import List, Tuple, Union, Optional

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    """
    MLP 基础网络模块
    作用：将输入特征通过多层感知机 (MLP) 进行特征提取, 并可选地进行特征归一化

    参数：
        args: argparse.Namespace
            包含超参数的命名空间, 需包括：
            - use_feature_normalization (bool): 是否使用 LayerNorm 进行特征归一化
            - use_orthogonal (bool): 是否使用正交初始化
            - use_ReLU (bool): 是否在隐藏层使用 ReLU 激活
            - layer_N (int): 隐藏层层数
            - hidden_size (int): 隐藏层维度
        input_dim: int
            MLP 输入特征的维度
    """
    def __init__(self, args: argparse.Namespace, input_dim: int) -> None:
        super(MLPBase, self).__init__()
        # 保存配置参数
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # 确定 MLP 的输入维度
        self.obs_dim = input_dim

        # 如果启用特征归一化，则创建 LayerNorm 模块
        if self._use_feature_normalization:
            # LayerNorm 的 normalized_shape 应与输入特征维度一致
            print(f"[DEBUG] LayerNorm normalized_shape = {self.obs_dim}")
            self.feature_norm = nn.LayerNorm(self.obs_dim)

        # 构建多层感知机主干，参数说明：
        #   - input_dim: 输入特征维度
        #   - hidden_size: 隐藏层维度
        #   - num_layers: 隐藏层数量
        #   - use_orthogonal: 权重初始化方式
        #   - use_ReLU: 是否使用 ReLU 激活
        self.mlp = MLPLayer(
            self.obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算

        参数：
            x: torch.Tensor
                输入特征张量，shape 为 [batch, ..., C]

        返回：
            torch.Tensor
                MLP 输出特征
        """
        # 调试：打印输入张量的形状
        print(f"[DEBUG] MLP forward input shape = {x.shape}")

        if self._use_feature_normalization:
            # 如果最后一维小于期望维度，则右侧用 0 填充到 obs_dim
            if x.shape[-1] < self.obs_dim:
                pad_size = self.obs_dim - x.shape[-1]
                pad = torch.zeros(*x.shape[:-1], pad_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
                print(f"[DEBUG] Padded input to shape: {x.shape}")
            # 如果最后一维大于期望维度，则截断多余部分
            elif x.shape[-1] > self.obs_dim:
                x = x[..., :self.obs_dim]
                print(f"[DEBUG] Truncated input to shape: {x.shape}")
            # 对特征进行 LayerNorm 归一化
            x = self.feature_norm(x)

        # 将归一化或原始特征传入 MLP 主干，输出最终特征
        return self.mlp(x)
