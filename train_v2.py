import torch
torch.cuda.empty_cache()  # 释放 GPU 缓存
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from envs.migratory_pgg_env_v3 import MigratoryPGGEnv  # 修改后的环境文件，观测为 (row, col)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from envs.visualizer import PygameVisualizer  # 导入可视化器
from datetime import datetime

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ———————————————————— 训练参数 ———————————————————————— #
TOTAL_STEPS = 1000000            # 总步数（环境步数）
update_interval = 1             # 每执行 k 步动作后更新一次 Q 表（可设置为 1 表示一步更新）
num_updates = TOTAL_STEPS // update_interval  # 更新次数
learning_rate = 0.1             # 学习率 α
gamma = 0.1                   # 折扣因子 γ
epsilon = 1.0                   # 初始探索率
epsilon_min = 1e-8              # 最小探索率
epsilon_decay = 0.999995          # 探索率衰减
seed = 42                       # 全局随机种子

# ———————————————————— 初始化 ———————————————————————— #
writer = SummaryWriter()        # TensorBoard 记录

# # 定义8×8区域环境的梯度方案参数：
# grid_division = 8
# custom_grid_params = {}

# # 设置参数取值
# r_min = 2.0   # 最左侧的增益因子最小值
# r_max = 5.0   # 最右侧的增益因子最大值
# c_min = 0.8   # 底部的固定成本最小值
# c_max = 1.6   # 顶部的固定成本最大值

# # 遍历每个网格单元，(i, j) 其中 i 代表行，j 代表列
# for i in range(grid_division):
#     for j in range(grid_division):
#         # 固定成本：从下到上线性增加
#         # 注意：这里假定 i=0 为顶部，i=7 为底部，所以用 (grid_division-1-i) 来实现从下到上增加
#         cost = c_min + ((grid_division - 1 - i) / (grid_division - 1)) * (c_max - c_min)
#         # 增益因子：从左到右线性增加
#         r = r_min + (j / (grid_division - 1)) * (r_max - r_min)
#         custom_grid_params[(i, j)] = (cost, r)

# # 定义8×8区域环境的山脉方案参数：
grid_division = 8
custom_grid_params = {}

for i in range(grid_division):
    for j in range(grid_division):
        if i in [3, 4] and j in [3, 4]:
            # Layer 1: 中央区（4个格子）
            cost = 0.8
            r = 5.0
        elif 2 <= i <= 5 and 2 <= j <= 5:
            # Layer 2: 中间环（12个格子）
            cost = 1.0
            r = 4.0
        elif 1 <= i <= 6 and 1 <= j <= 6:
            # Layer 3: 第三层（20个格子）
            cost = 1.2
            r = 3.5
        else:
            # Layer 4: 最外层（28个格子）
            cost = 1.6
            r = 3.0
        custom_grid_params[(i, j)] = (cost, r)

# 定义8×8区域环境的盆地方案参数：
# grid_division = 8
# custom_grid_params = {}

# for i in range(grid_division):
#     for j in range(grid_division):
#         if i in [3, 4] and j in [3, 4]:
#             # Layer 1: 中央区（4个格子）
#             cost = 1.6
#             r =  3.0
#         elif 2 <= i <= 5 and 2 <= j <= 5:
#             # Layer 2: 中间环（12个格子）
#             cost =  1.2
#             r =  3.5
#         elif 1 <= i <= 6 and 1 <= j <= 6:
#             # Layer 3: 第三层（20个格子）
#             cost = 1.0
#             r = 4.0  
#         else:
#             # Layer 4: 最外层（28个格子）
#             cost =  0.8
#             r = 5.0
#         custom_grid_params[(i, j)] = (cost, r)


# 实例化环境时传入自定义的区域参数
env = MigratoryPGGEnv(grid_division=8, custom_grid_params=custom_grid_params, seed=seed)
obs, _ = env.reset(seed=seed)

# 实例化可视化器对象，用于生成快照  
visualizer = PygameVisualizer(env)

# 在初始时刻（t=0）生成一次快照
visualizer.render(t=0)

# 初始化每个智能体的 Q 表：形状为 (network_size, network_size, 2, 5)
Q_tables = {
    agent: np.zeros((env.network_size, env.network_size, 2, 5))
    for agent in env.agent_names
}

# 初始化全局和区域合作率记录
# 注意：使用 env.grid_params 的 key 来表示各个网格区域
cooperation_history = {grid: [] for grid in env.grid_params.keys()}

# 如果采用批量更新，定义一个 transition 缓冲区，存储 (agent, s, action, reward, s')
transition_buffer = []

# -------------------------
# 数据保存相关设置
# 每 10,000 步写入一次 CSV 文件
SAVE_INTERVAL = 10000

# 在当前项目目录下的 results 文件夹中创建时间戳文件夹
results_root = "results"
os.makedirs(results_root, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(results_root, timestamp)
os.makedirs(results_dir, exist_ok=True)

# 在时间戳文件夹下创建 agent 和 grid 子文件夹
agent_dir = os.path.join(results_dir, "agent")
grid_dir = os.path.join(results_dir, "grid")
os.makedirs(agent_dir, exist_ok=True)
os.makedirs(grid_dir, exist_ok=True)

# 初始化数据缓存
# 对于智能体数据：字典 key: agent_id, value: list of dicts，每个 dict 为一行记录
agent_data = {agent: [] for agent in env.agent_names}
# 对于网格数据：字典 key: grid (tuple), value: list of dicts，每个 dict 为一行记录
grid_data = {grid: [] for grid in env.grid_params.keys()}

# -------------------------
# 预定生成快照的时间步（可根据需要调整）
snapshot_steps = {100, 1000, 10000, 100000, 200000, 300000, 400000, 500000} | set(range(510000, 1000001, 10000))

# ———————————————————— 训练循环 ———————————————————————— #
tbar = tqdm(range(1, TOTAL_STEPS + 1), desc="Training")
for step in tbar:
    actions = {}
    current_states = {}  # 记录每个智能体的当前状态，用于后续 Q 表更新
    # 为每个智能体选择动作（epsilon 贪婪）
    for agent in env.agent_names:
        s = obs[agent]  # 当前状态，形式为 (row, col)
        current_states[agent] = s
        # 从 Q 表中获得当前状态所有动作的 Q 值
        q_vals = Q_tables[agent][s[0], s[1]]
        if random.random() < epsilon:
            # 随机选择动作：game_action 取 0/1, move_action 取 0~4
            game_action = random.randint(0, 1)
            move_action = random.randint(0, 4)
        else:
            # 贪婪选择最大 Q 值对应的动作
            idx = np.unravel_index(np.argmax(q_vals, axis=None), q_vals.shape)
            game_action, move_action = idx
        actions[agent] = (game_action, move_action)
    
    # 执行动作，获得下一个状态和奖励
    new_obs, rewards, _, _, _ = env.step(actions)
    
    # 对每个智能体存储 (s, action, reward, s') 转移并更新 Q 表
    for agent in env.agent_names:
        s = current_states[agent]
        a = actions[agent]
        r = rewards[agent]
        s_next = new_obs[agent]
        # 直接更新 Q 表（update_interval == 1）
        old_val = Q_tables[agent][s[0], s[1], a[0], a[1]]
        next_max = np.max(Q_tables[agent][s_next[0], s_next[1]])
        Q_tables[agent][s[0], s[1], a[0], a[1]] = old_val + learning_rate * (r + gamma * next_max - old_val)
    
    # 更新 epsilon 探索率（衰减但不低于 epsilon_min）
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # 更新 TensorBoard 指标
    global_coop = env.get_global_cooperation_rate()  # 全局合作率
    avg_reward = np.mean(list(rewards.values()))       # 平均奖励
    writer.add_scalar("Global Cooperation Rate", global_coop, step)
    writer.add_scalar("Average Reward", avg_reward, step)
    writer.add_scalar("Epsilon", epsilon, step)
    
    # 更新每个网格的合作率记录
    grid_coop_rates = env.get_grid_cooperation_rates()
    for grid, coop_rate in grid_coop_rates.items():
        cooperation_history[grid].append(coop_rate if coop_rate is not None else 0)
    
    # 更新观测
    obs = new_obs
    
    # 保存智能体数据：遍历每个智能体，记录当前时间步下数据
    # 保存智能体数据时，直接保存位置数组和动作数组，并添加所属网格信息
    for agent in env.agent_names:
        s = current_states[agent]  # (row, col)
        grid_row = s[0] // env.grid_size
        grid_col = s[1] // env.grid_size
        agent_data[agent].append({
            't': step,
            'pos': [s[0], s[1]],
            'grid': [grid_row, grid_col],
            'action': actions[agent],
            'reward': rewards[agent]
        })

    # 保存网格数据：遍历每个网格，计算当前时间步网格内智能体数量及合作率
    for grid_key in env.grid_params.keys():
        grid_row, grid_col = grid_key
        # 计算当前网格内智能体数量
        count = sum(1 for agent in env.agents.values()
                    if agent.current_node[0] // env.grid_size == grid_row and
                       agent.current_node[1] // env.grid_size == grid_col)
        # 获取合作率（如果没有智能体，则记为 None）
        coop = grid_coop_rates.get(grid_key, None)
        grid_data[grid_key].append({
            't': step,
            'agent_count': count,
            'coop_rate': coop
        })
    
    # 每隔 SAVE_INTERVAL 步写入一次 CSV 文件
    if step % SAVE_INTERVAL == 0:
        # 写入智能体数据
        for agent, records in agent_data.items():
            df = pd.DataFrame(records)
            agent_file = os.path.join(agent_dir, f"{agent}.csv")
            # 如果文件已存在，则追加写入，不写表头
            if os.path.exists(agent_file):
                df.to_csv(agent_file, mode='a', index=False, header=False)
            else:
                df.to_csv(agent_file, index=False)
        # 写入网格数据
        for grid_key, records in grid_data.items():
            # 文件名采用 grid_{row}_{col}.csv
            grid_file = os.path.join(grid_dir, f"grid_{grid_key[0]}_{grid_key[1]}.csv")
            df = pd.DataFrame(records)
            if os.path.exists(grid_file):
                df.to_csv(grid_file, mode='a', index=False, header=False)
            else:
                df.to_csv(grid_file, index=False)
        # 清空缓存
        agent_data = {agent: [] for agent in env.agent_names}
        grid_data = {grid: [] for grid in env.grid_params.keys()}
    
    # 更新进度条显示
    tbar.set_postfix({
        "Global Coop": f"{global_coop:.3f}",
        "Avg Reward": f"{avg_reward:.3f}",
        "Epsilon": f"{epsilon:.3f}"
    })
    
    # 每隔 100 步打印一次指标
    if step % 100 == 0:
        print(f"Step {step}: Global Coop = {global_coop:.3f}, Avg Reward = {avg_reward:.3f}, Epsilon = {epsilon:.3f}")
    
    # 在指定快照时刻生成环境快照
    if step in snapshot_steps:
        visualizer.render(t=step)

writer.close()
