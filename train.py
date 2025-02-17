import torch
import numpy as np
from envs.migratory_pgg_env import MigratoryPGGEnv
from algorithms.DQNPolicy import DQNPolicy
from utils.replaybuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 训练参数
max_timesteps = 100000
learning_rate = 1e-5
buffer_size = 1000
batch_size = 16
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
gamma = 0.95

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境
env = MigratoryPGGEnv()
obs_spaces = env.get_observation_spaces(phase="game")  # 观测空间字典
action_spaces = {
    agent: {"game": env.action_spaces[agent], "move": env.move_action_spaces[agent]}
    for agent in env.agent_names
}

# 为每个智能体创建独立的 DQNPolicy
policies = {
    agent: DQNPolicy(action_spaces[agent], lr=learning_rate, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size, device=device)
    for agent in env.agent_names
}

# TensorBoard 记录
writer = SummaryWriter()

# 训练循环
obs, _ = env.reset()
prev_experiences = {agent: {"game": None, "move": None} for agent in env.agent_names}
reward = 0
loss = []
cooperation_rate = 0

progress_bar = tqdm(range(max_timesteps), desc="Training Progress", position=0, leave=True)

for t in progress_bar:
    actions = {}
    pre_game_obs = env.get_observation_spaces(env.phase)  # 记录 pre-game 状态

    # Game Phase: 贡献决策
    for agent in env.agent_names:
        actions[agent] = policies[agent].select_action(obs[agent], epsilon, phase="game")

    obs, rewards, _, _, _ = env.step(actions)  # o_G^t
    post_game_obs = env.get_observation_spaces(env.phase)  # 记录博弈后状态
    avg_reward = sum(rewards.values()) / len(env.agent_names)

    # 记录游戏经验，但不立即存入 buffer_G，需要等到下一个 pre-game 阶段
    for agent in env.agent_names:
        prev_experiences[agent]["game"] = (pre_game_obs[agent], actions[agent], rewards[agent], None, False)

    reward += sum(rewards.values())

    # Move Phase: 迁移决策
    for agent in env.agent_names:
        valid_moves = env.valid_actions(agent)
        actions[agent] = policies[agent].select_action(obs[agent], epsilon, phase="move", valid_moves=valid_moves)

    next_obs, rewards, _, _, _ = env.step(actions)  # o_M^t, R^t

    # 存储 Move Phase 经验
    for agent in env.agent_names:
        if prev_experiences[agent]["move"] is not None:
            policies[agent].store_experience(
                prev_experiences[agent]["move"][0],  # o_M^{t-1}
                prev_experiences[agent]["move"][1],  # a_M^{t-1}
                rewards[agent],                      # R^t
                post_game_obs[agent],                # o_M^t
                False,
                phase="move"
            )

        prev_experiences[agent]["move"] = (post_game_obs[agent], actions[agent], None, None, False)

    obs = next_obs  # 更新 obs
    reward += sum(rewards.values())

    # 存储 Game Phase 经验
    for agent in env.agent_names:
        if prev_experiences[agent]["game"] is not None:
            prev_state, prev_action, prev_reward, _, done = prev_experiences[agent]["game"]
            policies[agent].store_experience(prev_state, prev_action, prev_reward, pre_game_obs[agent], done, phase="game")

    # 训练 Q 网络
    losses = []
    for agent in env.agent_names:
        loss_g = policies[agent].train(phase="game")
        loss_m = policies[agent].train(phase="move")

        if loss_g is not None:
            losses.append(loss_g)
        if loss_m is not None:
            losses.append(loss_m)

    avg_loss = np.mean(losses) if losses else 0  # 计算平均 loss
    loss = []  # 重置 loss 数组，避免累积

    # 逐步衰减探索率
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    episode_cooperation_rate = env.coopration_rate()

    # 更新目标 Q 网络
    if t % 500 == 0:
        for agent in env.agent_names:
            policies[agent].update_target_network()

    # 记录进度信息
    if t % 50 == 0:
        writer.add_scalar("Avg Reward", avg_reward, t)
        writer.add_scalar("Loss", avg_loss, t)
        writer.add_scalar("Epsilon", epsilon, t)
        writer.add_scalar("Cooperation Rate", np.mean(episode_cooperation_rate) if episode_cooperation_rate else 0, t)

    # 更新 tqdm 进度条信息
    progress_bar.set_postfix({
        "Avg_Reward": f"{avg_reward:.2f}",
        "Loss": f"{avg_loss:.4f}",
        "Epsilon": f"{epsilon:.3f}",
        "Coop_Rate": f"{episode_cooperation_rate:.3f}"
    })

    # 控制台输出（减少日志量，每 500 步打印一次）
    if t % 50 == 0:
        print(f"\n[t {t:6d}] Avg Reward: {avg_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Coop Rate: {episode_cooperation_rate:.3f}")

    if t%20==0:
        env.render(t)

writer.close()
