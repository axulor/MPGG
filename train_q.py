import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import deque  # 引入双端队列
from envs.migratory_pgg_env import MigratoryPGGEnv
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录数据

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 训练参数
T = 100000  # 训练轮数
learning_rate = 0.1  # 学习率 α
gamma = 0.95  # 折扣因子 γ
epsilon = 1.0  # 初始探索率
epsilon_min = 0.05  # 最小探索率
epsilon_decay = 0.99995  # 探索率衰减
reward_window = 5  # 移动奖励计算的过去窗口大小

# 初始化 TensorBoard
writer = SummaryWriter("runs/Q-learning")

# 初始化环境
env = MigratoryPGGEnv()
past_rewards = {agent: deque(maxlen=reward_window) for agent in env.agent_names} # 记录过去 n 轮的博弈奖励，双端队列

# 初始化 Q 表，使用 PyTorch 张量，存储到 GPU
Q_G = {}  # 存储博弈 Q 值
Q_M = {}  # 存储移动 Q 值
for casino in env.casinos:
    Q_G[casino] = torch.zeros((2,), dtype=torch.float32, device=device)  # 2 个博弈动作（合作/背叛）
    Q_M[casino] = torch.zeros((5,), dtype=torch.float32, device=device)  # 5 个移动动作（上下左右不动）

pre_game_obs, _ = env.reset()


cooperation_rates = []  # 记录合作率

# 训练过程
for t in range(T):
    pre_game_obs, _ = env.reset() # 仅归位phase标签获取观测，并未重置智能体位置
    actions_G = {}
    actions_M = {}

    # 博弈阶段
    for agent in env.agent_names:
        casino = pre_game_obs[agent][:2]  # (c, alpha)
        if random.uniform(0, 1) < epsilon:
            actions_G[agent] = random.choice([0, 1])  # ε-greedy 探索
        else:
            actions_G[agent] = torch.argmax(Q_G[casino]).item()

    # 进行博弈，获取奖励
    _, rewards, _, _, _ = env.step(actions_G) # 奖励为字典格式

    # 记录博弈历史奖励
    for agent in env.agent_names:
        past_rewards[agent].append(rewards[agent]) 

    # 移动阶段
    for agent in env.agent_names:
        if random.uniform(0, 1) < epsilon:
            actions_M[agent] = random.choice([0, 1, 2, 3, 4])  # ε-greedy 探索
        else:
            actions_M[agent] = torch.argmax(Q_M[casino]).item()

    # 执行移动
    next_casino_state, _, _, _, _ = env.step(actions_M)

    # 更新博弈 Q-learning
    for agent in env.agent_names:
        action = actions_G[agent]
        reward = rewards[agent]
        next_casino = next_casino_state[agent][:2]

        # Q-learning 更新
        Q_G[casino][action] += learning_rate * (
            reward + gamma * torch.max(Q_G[next_casino]) - Q_G[casino][action]
        )

    # 计算移动奖励, 过去 n 轮博弈奖励的平均值
    for agent in env.agent_names:
        avg_past_reward = np.mean(past_rewards[agent])

        # 更新移动 Q-learning
        action = actions_M[agent]

        Q_M[casino][action] += learning_rate * (
            avg_past_reward + gamma * torch.max(Q_M[next_casino]) - Q_M[casino][action]
        )

    # 更新探索率
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 记录训练数据到 TensorBoard
    if t % 100 == 0:
        coop_rate = env.coopration_rate()
        cooperation_rates.append(coop_rate)
        writer.add_scalar("Cooperation Rate", coop_rate, t)
        writer.add_scalar("Epsilon", epsilon, t)
        print(f"t {t}, Cooperation Rate: {coop_rate:.4f}, Epsilon: {epsilon:.4f}")

        # 记录每个赌场的合作率
        for casino in env.casinos:
            casino_coopration_rate = env.each_coopration_rate(casino)

        # 可视化
        if t % 2000 == 0:
            env.render(t)


# 关闭 TensorBoard
writer.close()
