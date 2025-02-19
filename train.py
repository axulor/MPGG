import torch
torch.cuda.empty_cache()  # 释放 GPU 缓存

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from envs.migratory_pgg_env import MigratoryPGGEnv
from collections import deque  # 引入双端队列
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录数据

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ———————————————————— 训练参数 ———————————————————————— #
T = 100000  # 步数
learning_rate = 0.1  # 学习率 α
gamma = 0.1  # 折扣因子 γ
epsilon = 1.0  # 初始探索率
epsilon_min = 0.05  # 最小探索率
epsilon_decay = 0.995  # 探索率衰减
reward_window = 1  # 移动奖励计算的过去窗口大小


# ———————————————————— 初始化 ———————————————————————— #
writer = SummaryWriter() # 初始化 TensorBoard
env = MigratoryPGGEnv() # 初始化环境
past_rewards = {agent: deque(maxlen=reward_window) for agent in env.agent_names} # 记录过去 n 轮的博弈奖励，双端队列
Q_G = {}  # 存储博弈 Q 值
Q_M = {}  # 存储移动 Q 值
for casino in env.casinos:
    Q_G[casino] = torch.zeros((2,), dtype=torch.float32, device=device)  # 2 个博弈动作（合作/背叛）
    Q_M[casino] = torch.zeros((5,), dtype=torch.float32, device=device)  # 5 个移动动作（上下左右不动）

cooperation_rates = []  # 记录合作率
avg_past_rewards = {agent: 0 for agent in env.agent_names}  # 记录过去 n 轮的平均奖励
pbar = tqdm(range(T), desc="Training Progress", ncols=100) # 进度条


# ———————————————————— 训练循环 ———————————————————————— #
for t in pbar:
    obs, _ = env.reset_obs() # 重置观测
    actions_G = {}
    actions_M = {}

    # 博弈阶段（仅执行，不更新 Q-learning）
    for agent in env.agent_names:
        casino_state = obs[agent][:2]  # (c, alpha)
        if random.uniform(0, 1) < epsilon:
            actions_G[agent] = random.choice([0, 1])  # ε-greedy 探索
        else:
            actions_G[agent] = torch.argmax(Q_G[casino_state]).item()

    # 进行博弈，获取奖励
    next_obs, rewards, _, _, _ = env.step(actions_G)  # 赌场状态不变

    # 记录博弈历史奖励
    for agent in env.agent_names:
        past_rewards[agent].append(rewards[agent])

    # **移动阶段**
    for agent in env.agent_names:
        casino_state = next_obs[agent][:2]
        if random.uniform(0, 1) < epsilon:
            actions_M[agent] = random.choice([0, 1, 2, 3, 4])  # ε-greedy 探索
        else:
            actions_M[agent] = torch.argmax(Q_M[casino_state]).item()

    # 执行移动
    next_next_obs, _, _, _, _ = env.step(actions_M)  # 赌场状态 **此时改变**

    # **延迟更新 博弈 Q-learning**
    for agent in env.agent_names:
        casino_state = obs[agent][:2]  # 仍然是初始状态
        action = actions_G[agent]
        reward = rewards[agent]
        next_casino_state = next_next_obs[agent][:2]  # **使用移动后的赌场状态**

        # 更新博弈 Q-learning
        Q_G[casino_state][action] += learning_rate * (
            reward + gamma * torch.max(Q_G[next_casino_state]) - Q_G[casino_state][action]
        )

    # **更新移动 Q-learning**
    for agent in env.agent_names:
        avg_past_reward = np.mean(past_rewards[agent])

        casino_state = next_obs[agent][:2]  # **博弈后的赌场**
        action = actions_M[agent]
        next_casino_state = next_next_obs[agent][:2]  # **移动后的赌场**

        Q_M[casino_state][action] += learning_rate * (
            avg_past_reward + gamma * torch.max(Q_M[next_casino_state]) - Q_M[casino_state][action]
        )

    # **更新探索率**
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    coop_rate = env.coopration_rate()
    pbar.set_postfix({
    "T": t,
    "CoopRate": f"{coop_rate:.4f}",
    "Epsilon": f"{epsilon:.4f}"})

    # **记录训练数据到 TensorBoard**
    if t % 100 == 0:
        writer.add_scalar("Cooperation Rate", coop_rate, t)
        writer.add_scalar("Epsilon", epsilon, t)
        print(f"[Step {t}] Cooperation Rate: {coop_rate:.4f}, Epsilon: {epsilon:.4f}")

        for casino in env.casinos:
            casino_coopration_rate = env.each_coopration_rate(casino)
            writer.add_scalar(f"Cooperation Rate/{casino}", casino_coopration_rate, t)
            writer.add_scalar(f"Q_G_Defect/{casino}", Q_G[casino][0].item(), t)
            writer.add_scalar(f"Q_G_Coop/{casino}", Q_G[casino][1].item(), t)

            writer.add_scalar(f"Q_M_Up/{casino}", Q_M[casino][0].item(), t)
            writer.add_scalar(f"Q_M_Down/{casino}", Q_M[casino][1].item(), t)
            writer.add_scalar(f"Q_M_Left/{casino}", Q_M[casino][2].item(), t)
            writer.add_scalar(f"Q_M_Right/{casino}", Q_M[casino][3].item(), t)
            writer.add_scalar(f"Q_M_Stay/{casino}", Q_M[casino][4].item(), t)

        env.render(t)

# 关闭 TensorBoard
writer.close()
