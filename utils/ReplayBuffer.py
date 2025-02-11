import random
import numpy as np
import torch

class ReplayBuffer:
    """
    经验回放缓冲区 (Replay Buffer)
    用于存储训练数据，并随机采样用于训练

    :param buffer_size: 缓冲区大小
    :param batch_size: 训练时采样的 batch 大小
    """

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []  # 存储经验数据
        self.position = 0  # 记录当前存储位置

    def push(self, state, action, reward, next_state, done):
        """
        存储一个经验
        """
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size  # 维护循环存储

    def sample(self):
        """
        随机采样一个 batch
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        """
        返回当前存储的经验数量
        """
        return len(self.memory)
