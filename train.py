import time
import numpy as np
import torch
from algorithms.dqn import DQNAgent
from envs.migratory_pgg_env import MigratoryPGGEnv

# ✅ 打印 GPU 信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train.py] Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 初始化环境
env = MigratoryPGGEnv(L=9, l=3, r_min=1.2, r_max=5.0, N=18)

# 确保环境已正确初始化
state, _ = env.reset()

# ✅ 从 `state` 直接计算状态维度
sample_agent = env.agents[0]
state_dim = len(state[sample_agent])  # 确保状态正确
game_action_dim = env.action_space(sample_agent).n
move_action_dim = env.action_space(sample_agent).n

# 创建 DQN 代理
game_dqn_agent = DQNAgent(state_dim, game_action_dim, log_dir="logs/game_dqn/")
move_dqn_agent = DQNAgent(state_dim, move_action_dim, log_dir="logs/move_dqn/")

num_episodes = 100
update_target_every = 5

for episode in range(num_episodes):
    state, _ = env.reset()
    done = {agent: False for agent in env.agents}
    total_reward = 0

    while not all(done.values()):
        actions = {}

        if env.phase == "pre_game":
            for agent in env.agents:
                actions[agent] = game_dqn_agent.select_action(state[agent], [0, 1])
        else:
            for agent in env.agents:
                valid_moves = list(range(env.action_space(agent).n))
                actions[agent] = move_dqn_agent.select_action(state[agent], valid_moves)

        next_state, rewards, terminations, truncations, _ = env.step(actions)

        for agent in env.agents:
            if env.phase == "pre_game":
                game_dqn_agent.replay_buffer.add(state[agent], actions[agent], rewards[agent], next_state[agent], terminations[agent])
            else:
                move_dqn_agent.replay_buffer.add(state[agent], actions[agent], rewards[agent], next_state[agent], terminations[agent])

        state = next_state
        done = terminations
        total_reward += sum(rewards.values())

    game_dqn_agent.train(episode=episode)
    move_dqn_agent.train(episode=episode)

    if episode % update_target_every == 0:
        game_dqn_agent.update_target_model()
        move_dqn_agent.update_target_model()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # ✅ 记录总奖励
    game_dqn_agent.writer.add_scalar("Total Reward", total_reward, episode)

env.close()
game_dqn_agent.close()
move_dqn_agent.close()
