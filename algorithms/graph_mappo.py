import time
import numpy as np
import argparse
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from algorithms.utils.graph_buffer import GraphReplayBuffer
from algorithms.utils.util import get_grad_norm, huber_loss, mse_loss
from algorithms.utils.valuenorm import ValueNorm
from algorithms.utils.util import check
import torch.jit as jit
import torch.cuda.amp as amp

class GR_MAPPO:
    """
    Trainer class for Graph MAPPO to update policies.
    """

    def __init__(self, 
        args: argparse.Namespace, 
        gr_mappolicy: GR_MAPPOPolicy, 
        device = torch.device("cpu")
        ):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.gr_mappolicy = gr_mappolicy

        # TODO: 配置训练超参数
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # TODO: 初始化值函数归一化器（value_normalizer）
        if self._use_popart:
            self.value_normalizer = self.gr_mappolicy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, 
        values, 
        value_preds_batch, 
        return_batch, 
        # active_masks_batch,
        ):
        """
        TODO: 计算 critic 的 value loss(支持 clipped loss, huber, value norm 等)
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

        # if self._use_value_active_masks:
        #     value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        value_loss = value_loss.mean()


    @torch.cuda.amp.autocast
    def ppo_update(self, 
                sample:Tuple,  # 从 buffer 中采样的一组训练数据
                update_actor:bool=True # 是否执行 actor 网络的更新，默认 True
                ) -> Tuple[ Tensor, Tensor, 
                            Tensor, Tensor, 
                            Tensor, Tensor]:

        share_obs_batch, obs_batch, node_obs_batch, adj_batch, agent_id_batch, \
        share_agent_id_batch, rnn_states_batch, rnn_states_critic_batch, \
        actions_batch, value_preds_batch, return_batch, masks_batch, \
        active_masks_batch,old_action_log_probs_batch, adv_targ, \
        available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 得到 critic 的预测值、actor 的 log概率、策略熵
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                        obs_batch,
                                                        share_obs_batch,
                                                        node_obs_batch,
                                                        adj_batch,
                                                        agent_id_batch,
                                                        share_agent_id_batch,
                                                        actions_batch,
                                                        )
        
        # 重要性采样 + PPO 剪切策略 loss
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch) # 重要性采样比值
        surr1 = imp_weights * adv_targ # 期望项
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 
                            1.0 + self.clip_param) * adv_targ # 裁剪项
        
        # 直对 actior 网络做更新，直接对所有样本平均 loss
        policy_action_loss = -torch.sum(torch.min(surr1, surr2), 
                                            dim=-1, keepdim=True).mean()
        policy_loss = policy_action_loss
        self.policy.actor_optimizer.zero_grad() # 清零避免梯度累积
        st = time.time() # 记录当前时间戳
        # 反向传播更新 actor
        if update_actor:
            self.scaler.scale((policy_loss - dist_entropy * self.entropy_coef)).backward() # 这里有梯度缩放
        actor_backward_time = time.time() - st # 记录刚才反向传播花费的时间
        self.scaler.unscale_(self.policy.actor_optimizer) # 将之前缩放过的梯度恢复原始尺度
        # 是否使用 最大梯度裁剪（防止梯度爆炸）
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                                            self.policy.actor.parameters(), 
                                            self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        # 使用 AMP 执行 optimizer.step()，完成一次权重更新
        self.scaler.step(self.policy.actor_optimizer)

        # 对critic网络做更新
        value_loss = self.cal_value_loss(values, 
                                        value_preds_batch, 
                                        return_batch, 
                                        )
        self.policy.critic_optimizer.zero_grad()
        st = time.time()
        critic_loss = (value_loss * self.value_loss_coef)      
        self.scaler.scale(critic_loss).backward()
        critic_backward_time = time.time() - st
        self.scaler.unscale_(self.policy.critic_optimizer)
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                                                self.policy.critic.parameters(), 
                                                self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.scaler.step(self.policy.critic_optimizer)
        self.scaler.update()

        return (value_loss, critic_grad_norm, policy_loss, 
                dist_entropy, actor_grad_norm, imp_weights,
                actor_backward_time, critic_backward_time)

    def train(self, 
            buffer: GraphReplayBuffer, # TODO
            update_actor: bool=True, # 默认更新 actor 网络
            ): 

        if self._use_popart or self._use_valuenorm: # 是否使用值函数归一化
            advantages = buffer.returns[:-1] - \
                        self.value_normalizer.denormalize(
                                                buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1] # 无归一化是直接计算优势函数
        
        # 对优势函数 A_t 执行均值方差标准化
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # 记录训练指标
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0


        # 训练循环，ppo_epoch = 4 表示整个 buffer 会被使用 4 次，提升 sample efficiency
        for _ in range(self.ppo_epoch):
            st = time.time() # 记录开始的时间
            # 选择合适的采样方式
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, 
                                                        self.num_mini_batch, 
                                                        self.data_chunk_length) # TODO
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, 
                                                            self.num_mini_batch) #TODO
            else:
                data_generator = buffer.feed_forward_generator(advantages, 
                                                            self.num_mini_batch) # TODO
            
            # 对每个 mini-batch 执行一次完整的 actor-critic 更新
            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, \
                actor_grad_norm, imp_weights, actor_bt, critic_bt = self.ppo_update(sample, update_actor)
                

                # actor_backward_time += actor_bt
                # critic_backward_time += critic_bt
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

            # print(f'PPO epoch time: {time.time() - st}')
            # print(f'PPO epoch actor backward time: {actor_backward_time}')
            # print(f'PPO epoch critic backward time: {critic_backward_time}')

        num_updates = self.ppo_epoch * self.num_mini_batch # 总共执行的 ppo_update 次数

        for k in train_info.keys():
            train_info[k] /= num_updates # 对指标求平均

        return train_info    



    def prep_training(self):
        """
        设置模型为训练模式
        """
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        """
        设置模型为评估（推理）模式
        """
        self.policy.actor.eval()
        self.policy.critic.eval()
