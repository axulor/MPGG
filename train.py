import numpy as np
import random
from envs.migratory_pgg_env import MigratoryPGGEnv
from agents.agent import Agent

# 训练参数
epsilon_initial = 1.0
decay_rate = 0.995
epsilon_min = 0.05
gamma = 0.99
eta = 0.1
e_max = 1000  # 最大训练回合
t_max = 50  # 每个回合的最大时间步
buffer_size = 1000  # 经验池大小
batch_size = 32  # 经验回放批量大小

# 初始化环境
env = MigratoryPGGEnv()
agents = env.initialize_agents()
num_agents = len(agents)

# 初始化 Q 表和经验池
Q_G = {agent.agent_id: np.zeros((3, 2)) for agent in agents}  # 博弈Q表
Q_M = {agent.agent_id: np.zeros((4, 5)) for agent in agents}  # 移动Q表
buffer_G = []
buffer_M = []

epsilon = epsilon_initial

# 初始化前一个时间步的信息
prev_o_G = {agent.agent_id: None for agent in agents}
prev_a_G = {agent.agent_id: None for agent in agents}
prev_R = {agent.agent_id: None for agent in agents}
prev_o_M = {agent.agent_id: None for agent in agents}
prev_a_M = {agent.agent_id: None for agent in agents}

for e in range(e_max):
    observations, _ = env.reset()
    for t in range(t_max):
        game_experiences = []
        move_experiences = []

        # 并行博弈阶段
        for agent in agents:
            agent_id = agent.agent_id
            o_G = observations[agent_id]
            a_G = agent.choose_action(o_G, epsilon)
            R = env.get_reward(agent, a_G)

            if t > 0 and prev_o_G[agent_id] is not None:
                buffer_G.append((prev_o_G[agent_id], prev_a_G[agent_id], prev_R[agent_id], o_G))
                if len(buffer_G) > buffer_size:
                    buffer_G.pop(0)

            game_experiences.append((agent_id, o_G, a_G, R))
            prev_o_G[agent_id] = o_G
            prev_a_G[agent_id] = a_G
            prev_R[agent_id] = R

        # 并行移动阶段
        for agent in agents:
            agent_id = agent.agent_id
            o_M = observations[agent_id]
            a_M = agent.choose_action(o_M, epsilon)
            env.move_agents(agent, a_M)
            
            if t > 0 and prev_o_M[agent_id] is not None:
                buffer_M.append((prev_o_M[agent_id], prev_a_M[agent_id], prev_R[agent_id], o_M))
                if len(buffer_M) > buffer_size:
                    buffer_M.pop(0)

            move_experiences.append((agent_id, o_M, a_M))
            prev_o_M[agent_id] = o_M
            prev_a_M[agent_id] = a_M

        # 经验回放更新 Q_G
        if len(buffer_G) >= batch_size:
            batch = random.sample(buffer_G, batch_size)
            for o_G, a_G, R, o_G_next in batch:
                target_G = R + gamma * np.max(Q_G[agent_id][o_G_next])
                Q_G[agent_id][o_G, a_G] = (1 - eta) * Q_G[agent_id][o_G, a_G] + eta * target_G

        # 经验回放更新 Q_M
        if len(buffer_M) >= batch_size:
            batch = random.sample(buffer_M, batch_size)
            for o_M, a_M, R, o_M_next in batch:
                target_M = R + gamma * np.max(Q_M[agent_id][o_M_next])
                Q_M[agent_id][o_M, a_M] = (1 - eta) * Q_M[agent_id][o_M, a_M] + eta * target_M

        # 动态调整探索率
        epsilon = max(epsilon * decay_rate, epsilon_min)

    print(f"Episode {e+1}/{e_max} completed, Epsilon: {epsilon:.4f}")
