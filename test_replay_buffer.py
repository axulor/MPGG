import torch
import numpy as np
from utils.ReplayBuffer import ReplayBuffer

# ✅ 1. 创建 ReplayBuffer
buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

# ✅ 2. 生成伪造数据并存储
for _ in range(100):  # 存储 100 条经验
    state = np.random.randn(4)
    action = np.random.randint(0, 2)
    reward = np.random.randn()
    next_state = np.random.randn(4)
    done = np.random.choice([0, 1])
    buffer.push(state, action, reward, next_state, done)

print(f"✅ 当前缓冲区大小: {len(buffer)}")  # 应该输出 100

# ✅ 3. 采样 batch 进行训练
states, actions, rewards, next_states, dones = buffer.sample()

print("✅ 采样的状态 shape:", states.shape)  # (32, 4)
print("✅ 采样的动作 shape:", actions.shape)  # (32,)
print("✅ 采样的奖励 shape:", rewards.shape)  # (32,)
