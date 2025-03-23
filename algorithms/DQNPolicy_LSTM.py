import torch
import numpy as np
from envs.migratory_pgg_env_v0 import MigratoryPGGEnv
from algorithms.DQNPolicy import DQNPolicy
from utils.replaybuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 训练参数
max_timesteps = 10**6  # 运行超长时间步
learning_rate = 1e-5
buffer_size = 1000
batch_size = 16
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99999  # 逐步减少探索率
gamma = 0.92

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境
env = MigratoryPGGEnv()
obs_spaces = {agent: env.observation_spaces[agent] for agent in env.agent_names}
action_spaces = {agent: {"game": env.action_spaces[agent], "move": env.move_action_spaces[agent]} for agent in env.agent_names}

# 为每个智能体创建独立的 DQNPolicy
policies = {agent: DQNPolicy(action_spaces[agent], lr=learning_rate, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size, device=device) for agent in env.agent_names}

# TensorBoard 记录
writer = SummaryWriter()

# 训练循环（单 episode 长时间运行）
obs, _ = env.reset()
prev_experiences = {agent: {"game": None, "move": None} for agent in env.agent_names}
total_reward = 0
total_loss = []

tqdm_bar = tqdm(range(max_timesteps), desc="Training Progress")
for t in tqdm_bar:
    actions = {}

    # 记录 pre-game 状态 (o_G^{t-1})
    game_obs_snapshot = {agent: obs[agent] for agent in env.agent_names}

    # Game Phase: 贡献决策
    for agent in env.agent_names:
        actions[agent] = policies[agent].select_action(obs[agent], epsilon, phase="game")
    
    next_obs, rewards, _, _, _ = env.step(actions)  # o_G^t
    post_game_obs_snapshot = {agent: next_obs[agent] for agent in env.agent_names}  # 记录博弈后状态 (o_M^{t-1})

    # 记录游戏经验（延迟存入 buffer）
    for agent in env.agent_names:
        prev_experiences[agent]["game"] = (game_obs_snapshot[agent], actions[agent], rewards[agent], None, False)
    
    obs = next_obs
    total_reward += sum(rewards.values())

    # Move Phase: 迁移决策
    for agent in env.agent_names:
        valid_moves = env.valid_actions(agent)  # 计算合法动作
        actions[agent] = policies[agent].select_action(obs[agent], epsilon, phase="move", valid_moves=valid_moves)
    
    next_obs, rewards, _, _, _ = env.step(actions)  # o_M^t, R^t

    # 存储 Move Phase 经验
    for agent in env.agent_names:
        if prev_experiences[agent]["move"] is not None:
            policies[agent].store_experience(
                prev_experiences[agent]["move"][0],  # o_M^{t-1}
                prev_experiences[agent]["move"][1],  # a_M^{t-1}
                rewards[agent],                         # R^t
                post_game_obs_snapshot[agent],          # o_M^t
                False,
                phase="move"
            )
        prev_experiences[agent]["move"] = (post_game_obs_snapshot[agent], actions[agent], None, None, False)
    
    obs = next_obs  # 更新 obs
    total_reward += sum(rewards.values())

    # 存储 Game Phase 经验
    for agent in env.agent_names:
        if prev_experiences[agent]["game"] is not None:
            prev_state, prev_action, prev_reward, _, done = prev_experiences[agent]["game"]
            policies[agent].store_experience(prev_state, prev_action, prev_reward, game_obs_snapshot[agent], done, phase="game")

    # 训练 Q 网络
    for agent in env.agent_names:
        loss_g = policies[agent].train(phase="game")
        loss_m = policies[agent].train(phase="move")
        
        if loss_g is not None:
            total_loss.append(loss_g)
        if loss_m is not None:
            total_loss.append(loss_m)

    # 逐步衰减探索率
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    # 每 10k 步记录训练数据
    if t % 10000 == 0:
        cooperation_rate = env.coopration_rate()
        writer.add_scalar("Total Reward", total_reward, t)
        writer.add_scalar("Loss", np.mean(total_loss) if total_loss else 0, t)
        writer.add_scalar("Epsilon", epsilon, t)
        writer.add_scalar("Cooperation Rate", cooperation_rate, t)
        print(f"Step {t}: Reward={total_reward:.2f}, Loss={np.mean(total_loss) if total_loss else 0:.4f}, Epsilon={epsilon:.3f}, Cooperation Rate={cooperation_rate:.3f}")

    # 每 50k 步更新目标网络
    if t % 50000 == 0:
        for agent in env.agent_names:
            policies[agent].update_target_network()

writer.close()
