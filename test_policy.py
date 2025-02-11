import torch
import numpy as np
from gymnasium import spaces
from algorithms.DQNPolicy import DQNPolicy

# ✅ 1. 创建环境的观察空间 & 动作空间
obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
action_space = spaces.Discrete(2)

# ✅ 2. 创建 DQNPolicy
policy = DQNPolicy(obs_space, action_space)

# ✅ 3. 测试前向传播（计算 Q 值）
obs = torch.randn(1, 4)  # 随机观察值
q_values = policy.forward(obs)
print("✅ Q-values:", q_values)

# ✅ 4. 测试选择动作
selected_action = policy.select_action(obs.numpy(), epsilon=0.1)
print("✅ Selected action:", selected_action)

# ✅ 5. 生成伪造的 batch 数据（模拟经验回放）
batch_size = 32
batch = (
    torch.randn(batch_size, 4),  # states
    torch.randint(0, 2, (batch_size,)),  # actions
    torch.randn(batch_size,),  # rewards
    torch.randn(batch_size, 4),  # next_states
    torch.randint(0, 2, (batch_size,)).float(),  # dones
)

# ✅ 6. 训练 DQNPolicy
loss = policy.train(batch)
print("✅ Training Loss:", loss)

# ✅ 7. 更新目标网络
policy.update_target_network()
print("✅ Target network updated!")

