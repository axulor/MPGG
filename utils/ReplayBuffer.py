import random
import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size=32, device=None, prioritized=False, alpha=0.6, beta=0.4):
        """
        经验回放缓冲区，支持 Prioritized Experience Replay (PER)，默认行为与标准 DQN 兼容。
        :param buffer_size: 经验池大小
        :param batch_size: 每次训练采样的大小
        :param device: 运行设备
        :param prioritized: 是否使用优先经验回放 (默认 False)
        :param alpha: 经验优先级采样参数，仅在 prioritized=True 时生效
        :param beta: 重要性采样修正参数，仅在 prioritized=True 时生效
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        
        # 经验池
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size) if prioritized else None
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        存储一个经验 (s, a, r, s', done)，支持优先经验回放，默认行为不变。
        """
        self.memory.append((state, action, reward, next_state, done))
        if self.prioritized:
            priority = abs(td_error) + 1e-5 if td_error is not None else max(self.priorities, default=1.0)
            self.priorities.append(priority)
    
    def sample(self):
        """
        采样一个 batch，支持 PER，默认采用标准随机采样。
        """
        if self.prioritized and len(self.priorities) > 0:
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            weights = (len(self.memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            indices = random.sample(range(len(self.memory)), min(self.batch_size, len(self.memory)))
            weights = torch.ones(len(indices), dtype=torch.float32, device=self.device)
        
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        if self.prioritized:
            return (
                torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
                torch.tensor(actions, dtype=torch.int64).to(self.device),
                torch.tensor(rewards, dtype=torch.float32).to(self.device),
                torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
                torch.tensor(dones, dtype=torch.float32).to(self.device),
                weights, indices
            )
        else:
            return (
                torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
                torch.tensor(actions, dtype=torch.int64).to(self.device),
                torch.tensor(rewards, dtype=torch.float32).to(self.device),
                torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
                torch.tensor(dones, dtype=torch.float32).to(self.device)
            )
    
    def update_priorities(self, indices, td_errors):
        """
        更新优先级，基于 TD 误差，仅在 PER 机制下生效。
        """
        if self.prioritized and indices is not None:
            for idx, td_error in zip(indices, td_errors.cpu().numpy()):
                self.priorities[idx] = abs(td_error) + 1e-5
    
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
        if self.prioritized:
            self.priorities.clear()
    
    def advance_step(self):
        """
        维护 buffer_m 经验池，仅保留最新的移动经验
        """
        if len(self.memory) > self.buffer_size:
            self.memory.popleft()
            if self.prioritized:
                self.priorities.popleft()
