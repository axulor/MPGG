import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter  # ✅ 添加 TensorBoard

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, log_dir="logs/dqn/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")

        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer = ReplayBuffer(10000)

        # ✅ 添加 TensorBoard 记录
        self.writer = SummaryWriter(log_dir)

    def select_action(self, state_dict, valid_actions):
        """基于 DQN 选择合法动作"""
        state = torch.FloatTensor(np.array(state_dict)).to(self.device)

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = self.model(state).detach().cpu().numpy()
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values, key=valid_q_values.get)

    def train(self, batch_size=64, episode=0):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ✅ 记录 TensorBoard
        self.writer.add_scalar("Loss", loss.item(), episode)
        self.writer.add_scalar("Epsilon", self.epsilon, episode)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def close(self):
        """关闭 TensorBoard 记录器"""
        self.writer.close()
