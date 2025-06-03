import time
import numpy as np
import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from utils.graph_buffer import GraphReplayBuffer
from utils.util import get_grad_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check
import torch.jit as jit
import torch.cuda.amp as amp
class GR_MAPPO():
    """
        Trainer class for Graph MAPPO to update policies.
        args: (argparse.Namespace)  
            Arguments containing relevant model, policy, and env information.
        policy: (GR_MAPPO_Policy) 
            Policy to update.
        device: (torch.device) 
            Specifies the device to run on (cpu/gpu).
    """
    def __init__(self, 
                args:argparse.Namespace, 
                policy:GR_MAPPOPolicy,
                device=torch.device("cpu")) -> None:

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self.use_max_grad_norm = args.use_max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_huber_loss = args.use_huber_loss
        self.use_popart = args.use_popart
        self.use_valuenorm = args.use_valuenorm

        self.scaler = amp.GradScaler() 
        assert (self.use_popart and self.use_valuenorm) == False, ("self.use_popart and self.use_valuenorm can not be set True simultaneously")
        
        if self.use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self.use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, 
                    values:Tensor, 
                    old_values_batch:Tensor, 
                    returns_batch:Tensor) -> Tensor:
        """
        Return: 
        - value_loss: 单个标量 loss, 用于反向传播更新价值网络
        """
        # 计算 PPO 价值裁剪 (Value Clipping) 中的裁剪后的价值预测
        value_pred_clipped = old_values_batch + (values - 
                            old_values_batch).clamp(-self.clip_param,
                                                    self.clip_param)
        if self.use_popart or self.use_valuenorm: 
            self.value_normalizer.update(returns_batch)
            error_clipped = self.value_normalizer.normalize(returns_batch) - \
                            value_pred_clipped
            error_original = self.value_normalizer.normalize(returns_batch) - \
                            values
        else:
            error_clipped = returns_batch - value_pred_clipped
            error_original = returns_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss
    
    def ppo_update(self, sample:Tuple):
        """
        Return:
        - value_loss:          价值损失
        - critic_grad_norm:    Critic 网络梯度范数
        - policy_loss:         策略损失
        - dist_entropy:        策略熵
        - actor_grad_norm:     Actor 网络梯度范数
        - imp_weights:         重要性采样权重
        """
        share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
        share_agent_id_batch, \
        actions_batch, old_values_batch, returns_batch, \
        old_action_log_probs_batch, adv_targ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        old_values_batch = check(old_values_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)

        # 使用当前策略网络评估动作
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                        share_obs_batch,
                                                        obs_batch,
                                                        node_obs_batch,
                                                        adj_batch,
                                                        agent_id_batch,
                                                        share_agent_id_batch,
                                                        actions_batch)
        # 重要性采样权重
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ # 没有PPO截断
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 
                            1.0 + self.clip_param) * adv_targ # PPO截断
        # 策略损失
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()        
        # Actor 网络的梯度清零、反向传播和参数更新
        self.policy.actor_optimizer.zero_grad()
        actor_total_loss = policy_loss - dist_entropy * self.entropy_coef # 这是 Actor 要最小化的总损失 
        self.scaler.scale(actor_total_loss).backward() # 反向传播
        self.scaler.unscale_(self.policy.actor_optimizer)
        if self.use_max_grad_norm: # 启用梯度裁剪
            actor_grad_norm = nn.utils.clip_grad_norm_(
                                            self.policy.actor.parameters(), 
                                            self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.scaler.step(self.policy.actor_optimizer) # 更新 Actor 的参数

        # 价值损失
        value_loss = self.cal_value_loss(values,         # V_new(s_t)
                                        old_values_batch,   # V_old(s_t)
                                        returns_batch)   # G_t
        # Critic 网络的梯度清零、反向传播和参数更新
        self.policy.critic_optimizer.zero_grad()
        critic_loss = (value_loss * self.value_loss_coef)    # Critic 要最小化的总损失
        self.scaler.scale(critic_loss).backward() # 反向传播
        self.scaler.unscale_(self.policy.critic_optimizer) 
        if self.use_max_grad_norm: # 启用梯度裁剪
            critic_grad_norm = nn.utils.clip_grad_norm_(
                                                self.policy.critic.parameters(), 
                                                self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.scaler.step(self.policy.critic_optimizer) # 更新 Actor 的参数
        
        self.scaler.update()

        return (value_loss, critic_grad_norm, policy_loss, 
                dist_entropy, actor_grad_norm, imp_weights)

    def train(self, buffer:GraphReplayBuffer):
        """
        Return: 
            train_infos: 包含训练更新信息的字典
        """
        # 计算优势
        if self.use_popart or self.use_valuenorm:
            advantages = buffer.returns[:-1] - \
                        self.value_normalizer.denormalize(
                                                buffer.values[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.values[:-1]
        # 优势归一化
        mean_advantages = np.nanmean(advantages)
        std_advantages = np.nanstd(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_infos = {}
        train_infos['value_loss'] = 0        # 价值损失, value_loss, 逐渐减小并收敛
        train_infos['policy_loss'] = 0       # 策略损失, 逐渐下降或在一个范围内波动
        train_infos['dist_entropy'] = 0      # 策略熵, 逐渐下降（表示策略从探索到利用），但不能降为 0
        train_infos['actor_grad_norm'] = 0   # Actor 梯度范数, 应在一个合理的范围内
        train_infos['critic_grad_norm'] = 0  # Critic 梯度范数, 应在一个合理的范围内
        train_infos['imp_weights'] = 0       # 新策略选择动作 a_t 的概率与旧策略选择同样动作的概率之比, 在 1.0 附近波动

        for _ in range(self.ppo_epoch):
            # 将整个 Buffer 的数据分割成 self.num_mini_batch 个小批次
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, \
                actor_grad_norm, imp_weights = self.ppo_update(sample) # 用这批样本做一次网络更新
                
                train_infos['value_loss'] += value_loss.item()
                train_infos['policy_loss'] += policy_loss.item()
                train_infos['dist_entropy'] += dist_entropy.item()
                train_infos['actor_grad_norm'] += actor_grad_norm
                train_infos['critic_grad_norm'] += critic_grad_norm
                train_infos['imp_weights'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_infos.keys():
            train_infos[k] /= num_updates # 将累加值除以总更新次数得到平均值

        return train_infos

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_evaluating(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()
