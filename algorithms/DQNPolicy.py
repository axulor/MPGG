import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np
from gymnasium import spaces
from utils.replaybuffer import ReplayBuffer

class DQNPolicy:
    def __init__(self,  action_spaces, 
                lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=32, device="cpu"):
        self.device = device
        self.action_spaces = action_spaces  # 存储不同阶段的动作空间
        self.gamma = gamma
        self.batch_size = batch_size

        # 预定义观测空间维度
        self.obs_dim_g = 3  # 博弈前观测空间维度
        self.obs_dim_m = 4  # 博弈后观测空间维度

        # 贡献决策网络 (game) - 观测维度固定 3，动作维度固定 2
        self.q_g = self.build_q_network(3, 2)
        self.target_q_g = self.build_q_network(3, 2)

        # 迁移决策网络 (move) - 观测维度固定 4，动作维度固定 5
        self.q_m = self.build_q_network(4, 5)
        self.target_q_m = self.build_q_network(4, 5)

        # 初始化经验回放缓冲区
        self.buffer_g = ReplayBuffer(buffer_size,  batch_size=batch_size, device=self.device)
        self.buffer_m = ReplayBuffer(buffer_size,  batch_size=batch_size, device=self.device)

        # 优化器
        self.optimizer_g = torch.optim.Adam(self.q_g.parameters(), lr=lr)
        self.optimizer_m = torch.optim.Adam(self.q_m.parameters(), lr=lr)
    
    def build_q_network(self, obs_dim, action_dim):
        """ 创建 Q-Network 并初始化参数 """
        model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)

        # 参数初始化
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier 初始化
                layer.bias.data.fill_(0.01)  # 初始化偏置
        return model

    def select_action(self, obs, epsilon, phase="game", valid_moves=None):
        """ 选择动作，确保 move 阶段只选择合法动作 """
        if np.random.rand() < epsilon:
            if phase == "move" and valid_moves is not None:
                return np.random.choice(valid_moves)  # 只选取合法动作
            return np.random.randint(self.action_spaces[phase].n)
        else:
            obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_g(obs_tensor) if phase == "game" else self.q_m(obs_tensor)
                if phase == "move" and valid_moves is not None and len(valid_moves) > 0:
                    q_values = q_values[:, valid_moves]  # 只考虑合法动作
                    return valid_moves[q_values.argmax().item()]
                return q_values.argmax().item()


    def store_experience(self, state, action, reward, next_state, done, phase="game"):
        """ 存储经验到 buffer_g 或 buffer_m """
        if phase == "game":
            self.buffer_g.add(state, action, reward, next_state, done)
        else:
            self.buffer_m.add(state, action, reward, next_state, done)

    def train(self, phase="game", valid_moves=None):
        buffer = self.buffer_g if phase == "game" else self.buffer_m
        q_net = self.q_g if phase == "game" else self.q_m
        target_q_net = self.target_q_g if phase == "game" else self.target_q_m
        optimizer = self.optimizer_g if phase == "game" else self.optimizer_m

        if len(buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = buffer.sample()


        q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = target_q_net(next_states)
            if phase == "move" and valid_moves is not None and len(valid_moves) > 0:
                next_q_values = next_q_values[:, valid_moves]
            max_next_q = next_q_values.max(dim=1)[0]

        q_targets = rewards + (1 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_values, q_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()




    def update_target_network(self):
        """ 更新目标 Q 网络 """
        self.target_q_g.load_state_dict(self.q_g.state_dict())
        self.target_q_m.load_state_dict(self.q_m.state_dict())
