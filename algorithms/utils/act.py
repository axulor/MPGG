"""
Modules to compute actions. Basically the final layer in the network.
This imports modified probability distribution layers wherein we can give action
masks to re-normalise the probability distributions
"""
from .distributions import Categorical, DiagGaussian
import torch
import torch.nn as nn
from typing import Optional


class ACTLayer(nn.Module):
    """
    用于计算动作的MLP模块
    
    action_space:       Gym 连续空间对象;
    inputs_dim:         网络输入张量的特征维度;
    use_orthogonal:     权重初始化为正交初始化;
    gain:               网络输出层的增益;
    """

    def __init__(
        self, action_space, inputs_dim: int, use_orthogonal: bool, gain: float
    ):
        super(ACTLayer, self).__init__()

        action_dim = action_space.shape[0] # 动作空间维度
        self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain) # 动作分布类

        
    def forward(
        self,
        x: torch.tensor,
        deterministic: bool = False,
    ):
        """
        Return:
            actions: torch.Tensor
                要执行的动作
            action_log_probs: torch.Tensor
                动作的对数概率
        """

        action_logits = self.action_out(x)        # 将数据送入分布类 
        actions = action_logits.mode() if deterministic else action_logits.sample() # 从分布类中采样动作向量，2维
        action_log_probs = action_logits.log_probs(actions) # 二维动作向量计算成对数概率 


        return actions, action_log_probs

    def evaluate_actions(
        self,
        x: torch.tensor,
        action: torch.tensor,
    ):
        """
        Rerurn:
            action_log_probs: 对应输入动作的 log π(a|s)        
            dist_entropy:     分布的平均熵，用于鼓励探索
                
        """
        
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
