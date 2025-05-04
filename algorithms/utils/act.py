"""
Modules to compute actions. Basically the final layer in the network.
This imports modified probability distribution layers wherein we can give action
masks to re-normalise the probability distributions
"""
from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn
from typing import Optional


class ACTLayer(nn.Module):
    """
    用于计算动作的MLP模块
    
    action_space: Gym 空间对象，描述动作维度和类型（仅支持离散、连续）
    inputs_dim: 网络输入张量的特征维度, int
    use_orthogonal: 控制输出层权重初始化方式, bool
    gain: 网络输出层的增益, float
    """

    def __init__(
        self, action_space, inputs_dim: int, use_orthogonal: bool, gain: float
    ):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        else:  
            print(f"错误: 暂不支持的动作空间类型 {action_space.__class__.__name__}")
            raise NotImplementedError
        
    def forward(
        self,
        x: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        deterministic: bool = False,
    ):
        """
        计算给定输入下的动作及其对数概率。

        参数：
            x: torch.Tensor
                网络的输入张量。
            available_actions: Optional[torch.Tensor]
                告知哪些动作对智能体是可用的（默认所有动作都可用）。
            deterministic: bool
                是否直接返回分布的最优模式(True), 否则根据分布进行采样False

        返回：
            actions: torch.Tensor
                要执行的动作。
            action_log_probs: torch.Tensor
                所选动作的对数概率。
        """
        # print(f"x:\ttype={type(x)}, shape={x.shape}, dtype={x.dtype}")
        # print(f"x[0]:\ttype={type(x[0])}, shape={x[0].shape}, dtype={x[0].dtype}")
        action_logits = self.action_out(x)        # 将数据送入分布类 
        # print(action_logits) 
        actions = action_logits.mode() if deterministic else action_logits.sample() # 从分布类中采样动作向量，2维
        action_log_probs = action_logits.log_probs(actions) # 二维动作向量计算成对数概率
        # print(f"actions:\ttype={type(actions)}, shape={actions.shape}, dtype={actions.dtype}")
        # print(f"actions[0]:\ttype={type(actions[0])}, shape={actions[0].shape}, dtype={actions[0].dtype}")  
        # print(f"action_log_probs:\ttype={type(action_log_probs)}, shape={action_log_probs.shape}, dtype={action_log_probs.dtype}")
        # print(f"action_log_probs[0]:\ttype={type(action_log_probs[0])}, shape={action_log_probs[0].shape}, dtype={action_log_probs[0].dtype}")  


        return actions, action_log_probs


    def evaluate_actions(
        self,
        x: torch.tensor,
        action: torch.tensor,
        available_actions: Optional[torch.tensor] = None,
        active_masks: Optional[torch.tensor] = None,
    ):
        """
        计算给定动作的对数概率和分布熵。

        参数：
        x:                  网络输入特征, shape=[batch, inputs_dim]
        action:             要评估的动作, shape 跟前面采样时一致
        available_actions:  可选的动作掩码（哪些动作合法），目前代码中并未使用
        active_masks:       掩码标识哪些样本“仍在跑”(1)或“已结束/死亡”(0), 用于熵计算时排除无效 agent

        返回：
        action_log_probs: torch.Tensor [batch, 1] 或 [batch, num_actions] （取决于分支）
            对应输入动作的 log π(a|s)
        dist_entropy:     torch.Tensor 标量或 [num_subspaces]
            分布的平均熵，用于鼓励探索
        """
        
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)


        dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
