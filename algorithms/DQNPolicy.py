import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from utils.ReplayBuffer import ReplayBuffer

class DQNPolicy:
    """
    DQN 策略类，包含 Q-Network、目标网络 和 经验回放
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete, buffer_size=10000, batch_size=32):
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size

        # 初始化 Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # 初始化 Target Q-Network
        self.target_q_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

    def forward(self, obs: torch.Tensor):
        """计算 Q 值"""
        return self.q_network(obs)

    def select_action(self, obs, epsilon=0.1):
        """Epsilon-greedy 选择动作"""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_space.n, (1,)).item()
        else:
            return self.forward(torch.tensor(obs, dtype=torch.float32)).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到 ReplayBuffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        """从 ReplayBuffer 采样并训练 Q-Network"""
        if len(self.replay_buffer) < self.batch_size:
            return  # 如果经验不足，先不训练

        # 采样 batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # 计算 Q(s, a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算 Q 目标值
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(dim=1)[0]
            q_targets = rewards + (1 - dones) * 0.99 * next_q_values  # γ=0.99

        # 计算损失并优化
        loss = F.smooth_l1_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标 Q 网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
