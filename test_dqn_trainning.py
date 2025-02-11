import torch
import numpy as np
from gymnasium import spaces
from algorithms.DQNPolicy import DQNPolicy

# 创建环境的观察空间 & 动作空间
obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
action_space = spaces.Discrete(2)

# 创建 DQNPolicy
policy = DQNPolicy(obs_space, action_space)

# 生成训练数据并存储
for _ in range(100):  # 存储 100 条经验
    state = np.random.randn(4)
    action = np.random.randint(0, 2)
    reward = np.random.randn()
    next_state = np.random.randn(4)
    done = np.random.choice([0, 1])
    policy.store_experience(state, action, reward, next_state, done)

print(f"当前缓冲区大小: {len(policy.replay_buffer)}")  # 应该输出 100

# 训练 500 步
for i in range(500):
    loss = policy.train()
    if loss is not None and i % 50 == 0:
        print(f"Step {i}, Loss: {loss:.4f}")

# 更新目标网络
policy.update_target_network()
print("Target network updated!")
