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

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        self.scaler = amp.GradScaler() 
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, 
                    values:Tensor, 
                    value_preds_batch:Tensor, 
                    return_batch:Tensor) -> Tensor:
        """
        计算价值网络的损失。
        - values: 当前网络对状态值 V(s) 的预测, shape=(batch,1) 或 (batch,)
        - value_preds_batch: 旧网络在采样时对同一批次状态的预测，用于“值函数截断”损失
        - return_batch: “回报”或“优势加价值”计算后得到的真实目标 return, shape=(batch,1)
        - active_masks_batch: mask 掩码, 标记哪些样本是有效(agent仍存活), 哪些是无效(已done)
        返回：
            单个标量 loss, 用于反向传播更新价值网络
        """
        value_pred_clipped = value_preds_batch + (values - 
                            value_preds_batch).clamp(-self.clip_param,
                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm: 
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - \
                            value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - \
                            values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original



        value_loss = value_loss.mean()

        return value_loss
    
    def ppo_update(self, 
                sample:Tuple, 
                update_actor:bool=True) -> Tuple[Tensor, Tensor, 
                                                Tensor, Tensor, 
                                                Tensor, Tensor]:
        """
        使用一个小批次数据对 Actor 和 Critic 同时做一次 PPO 更新
        
        返回：
        value_loss          价值函数损失（未乘 coef）
        critic_grad_norm    Critic 网络梯度范数
        policy_loss         策略（Actor）损失
        dist_entropy        策略输出的熵（鼓励探索）
        actor_grad_norm     Actor 网络梯度范数
        imp_weights         重要性采样权重 π/π_old
        actor_backward_time Actor 反向传播耗时
        critic_backward_time Critic 反向传播耗时
        """
        share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
        share_agent_id_batch, \
        actions_batch, value_preds_batch, return_batch, \
        old_action_log_probs_batch, adv_targ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        # print("MaPPO", active_masks_batch.T)
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                        share_obs_batch,
                                                        obs_batch,
                                                        node_obs_batch,
                                                        adj_batch,
                                                        agent_id_batch,
                                                        share_agent_id_batch,
                                                        actions_batch)
        # actor update
        # print(f'obs: {obs_batch.shape}')
        # st = time.time()
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 
                            1.0 + self.clip_param) * adv_targ
        # print(f'Surr1: {surr1.shape} \t Values: {values.shape}')


        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        # print(f'Actor Zero grad time: {time.time() - st}')
        st = time.time()

        if update_actor:
            # (policy_loss - dist_entropy * self.entropy_coef).backward()
            self.scaler.scale((policy_loss - dist_entropy * self.entropy_coef)).backward()
        critic_backward_time = time.time() - st
        # print(f'Actor Backward time: {critic_backward_time}')
        # st = time.time()
        self.scaler.unscale_(self.policy.actor_optimizer)
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                                            self.policy.actor.parameters(), 
                                            self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())

        # self.policy.actor_optimizer.step()
        self.scaler.step(self.policy.actor_optimizer) #.step()
        # print(f'Actor Step time: {time.time() - st}')
        # st = time.time()

        # critic update
        # print(values.shape, value_preds_batch.shape)
        value_loss = self.cal_value_loss(values, 
                                        value_preds_batch, 
                                        return_batch)

        self.policy.critic_optimizer.zero_grad()
        # print(f'Critic Zero grad time: {time.time() - st}')
        
        st = time.time()
        critic_loss = (value_loss * self.value_loss_coef)   # TODO add gradient accumulation here
        # critic_loss.backward()
        self.scaler.scale(critic_loss).backward()
        actor_backward_time = time.time() - st
        # print(f'Critic Backward time: {actor_backward_time}')
        # st = time.time()
        self.scaler.unscale_(self.policy.critic_optimizer)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                                                self.policy.critic.parameters(), 
                                                self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())

        # self.policy.critic_optimizer.step()
        self.scaler.step(self.policy.critic_optimizer) #.step()
        self.scaler.update()
        # print(f'Critic Step time: {time.time() - st}')
        # print('_'*50)

        return (value_loss, critic_grad_norm, policy_loss, 
                dist_entropy, actor_grad_norm, imp_weights,
                actor_backward_time, critic_backward_time)

    def train(self, 
            buffer:GraphReplayBuffer, 
            update_actor:bool=True):
        """
            Perform a training update using minibatch GD.
            使用小批量梯度下降执行一次训练更新。

            buffer: (GraphReplayBuffer)
                buffer containing training data.
                包含训练数据的缓冲区。
            update_actor: (bool)
                whether to update actor network.
                是否更新 Actor 网络。

            :return train_info: (dict)
                contains information regarding
                training update (e.g. loss, grad norms, etc).
                返回一个包含训练更新信息的字典（例如损失、梯度范数等）。
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - \
                        self.value_normalizer.denormalize(
                                                buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        mean_advantages = np.nanmean(advantages)
        std_advantages = np.nanstd(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0        # 价值损失累加
        train_info['policy_loss'] = 0       # 策略损失累加
        train_info['dist_entropy'] = 0      # 策略熵累加
        train_info['actor_grad_norm'] = 0   # Actor 梯度范数累加
        train_info['critic_grad_norm'] = 0  # Critic 梯度范数累加
        train_info['ratio'] = 0             # 重要性采样比率的平均值累加

        for _ in range(self.ppo_epoch):
            # self.ppo_epoch 指定重复更新的次数
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            
            # actor_backward_time, critic_backward_time = 0, 0 

            for sample in data_generator:
                # data_generator 会将整个 Buffer 的数据分割成 self.num_mini_batch 个小批次
                value_loss, critic_grad_norm, policy_loss, dist_entropy, \
                actor_grad_norm, imp_weights, actor_bt, critic_bt = self.ppo_update(sample, update_actor)
                
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()


        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates # 将累加值除以总更新次数得到平均值

        return train_info

    def prep_training(self):
        """Convert networks to training mode"""
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """Convert networks to eval mode"""
        self.policy.actor.eval()
        self.policy.critic.eval()
