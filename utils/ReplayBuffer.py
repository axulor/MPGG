import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size=32, device=None):
        """
        经验回放缓冲区
        :param buffer_size: 经验池大小
        :param batch_size: 每次训练采样的大小
        :param device: 运行设备
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # 经验池
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """
        存储一个经验 (s, a, r, s', done)
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        """
        随机采样一个 batch
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.int64).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )
    
    def __len__(self):
        """
        返回当前存储的经验数量
        """
        return len(self.memory)

    def clear(self):
        """
        清空经验池
        """
        self.memory.clear()
    
    def advance_step(self):
        """
        维护 buffer_m 经验池，仅保留最新的移动经验
        """
        if len(self.memory) > self.buffer_size:
            self.memory.popleft()
