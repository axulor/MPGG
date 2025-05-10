# --- START OF FILE marl_env.py (修改版) ---

import gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
# from scipy import sparse # 在当前 MPGG 实现中似乎未使用
import argparse # 确保导入，即使 __init__ 中用的是 SimpleNamespace
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# == Agent Class Definition (保持不变) ==
# ==============================================================================
class Agent:
    def __init__(self):
        self.id = None
        self.name = None
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.strategy = np.array([0], dtype=np.int32)
        self.last_payoff = np.array([0.0], dtype=np.float32)
        self.action = np.zeros(2, dtype=np.float32)

# ==============================================================================
# == Merged MultiAgentGraphEnv Class Definition (修改版) ==
# ==============================================================================
class MultiAgentGraphEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, args: argparse.Namespace): # 参数类型改为 argparse.Namespace 以匹配原始定义
        super().__init__()
        # --- 1. 参数初始化 (基本不变) ---
        self.num_agents = args.num_agents
        self.world_size = args.world_size
        self.speed = args.speed
        self.max_cycles = args.max_cycles # 环境内在上限
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r
        self.beta = args.beta
        self.discrete_action = args.discrete_action
        # self.scenario_name = args.scenario_name # 如果 train_mpgg.py 中不再定义，这里也不需要

        # --- 2. 初始化智能体列表 (不变) ---
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # --- 3. 初始化环境状态 ---
        self.current_step_in_episode = 0 # 当前 episode 内部的步数

        # --- 新增：用于累积 episode 级别指标的属性 ---
        self.episode_total_rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_cooperation_counts = np.zeros(self.num_agents, dtype=np.int32) # 记录每个 agent 合作的次数
        self.episode_steps_count = 0 # 记录当前 episode 实际运行的步数 (可能小于 max_cycles)

        # --- 4. 图和距离属性 (不变) ---
        self.edge_list = None
        self.edge_weight = None
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        self.cache_dists = True

        # --- 5. 配置 Gym 空间 (基本不变) ---
        if self.discrete_action:
            self.agent_action_space = spaces.Discrete(5)
        else:
            self.agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = [self.agent_action_space] * self.num_agents

        _temp_agent = Agent() # 用于获取特征维度
        # _temp_agent的属性需要与实际Agent一致，以获取正确的obs_dim
        _temp_agent.pos = np.zeros(2); _temp_agent.vel = np.zeros(2)
        _temp_agent.strategy = np.array([0]); _temp_agent.last_payoff = np.array([0.0])
        _obs_sample = self._get_agent_feat(_temp_agent) # _get_agent_feat 不再包含实体ID
        obs_dim = _obs_sample.shape[0] # 现在是 6 维

        self.agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = [self.agent_observation_space] * self.num_agents

        share_obs_dim = obs_dim * self.num_agents
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
        ] * self.num_agents

        node_obs_dim_tuple = (self.num_agents, obs_dim) # Gym space shape is tuple
        adj_dim_tuple = (self.num_agents, self.num_agents)
        agent_id_dim_tuple = (1,)
        edge_dim_tuple = (1,) # 假设边特征是标量（例如距离）

        self.node_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=node_obs_dim_tuple, dtype=np.float32)] * self.num_agents
        self.adj_observation_space = [spaces.Box(low=0, high=+np.inf, shape=adj_dim_tuple, dtype=np.float32)] * self.num_agents
        self.agent_id_observation_space = [spaces.Box(low=0, high=self.num_agents - 1, shape=agent_id_dim_tuple, dtype=np.int32)] * self.num_agents
        self.share_agent_id_observation_space = [
            spaces.Box(low=0, high=self.num_agents - 1, shape=(self.num_agents * agent_id_dim_tuple[0],), dtype=np.int32)
        ] * self.num_agents
        self.edge_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=edge_dim_tuple, dtype=np.float32)] * self.num_agents

    def seed(self, seed=None):
        np.random.seed(seed) # Gym 标准推荐这样设置

    def reset(self) -> Tuple[List[arr], List[arr], List[arr], List[arr]]:
        self.current_step_in_episode = 0 # 重置 episode 内部步数

        # --- 重置累加器 ---
        self.episode_total_rewards.fill(0.0)
        self.episode_cooperation_counts.fill(0)
        self.episode_steps_count = 0
        # --------------------

        shuffled_agents = list(self.agents)
        np.random.shuffle(shuffled_agents)
        half = (self.num_agents + 1) // 2
        cooperators = set(shuffled_agents[:half])

        for agent in self.agents:
            agent.pos = np.random.rand(2) * self.world_size
            theta = np.random.rand() * 2 * np.pi
            agent.vel = self.speed * np.array([np.cos(theta), np.sin(theta)])
            agent.last_payoff = np.array([0.0], dtype=np.float32)
            agent.strategy = np.array([1 if agent in cooperators else 0], dtype=np.int32)
            agent.action = np.zeros(2, dtype=np.float32)

        self.calculate_distances()
        self.update_graph() # 更新图结构以获取正确的初始 adj

        obs_n = [self._get_obs(agent) for agent in self.agents]
        agent_id_n = [self._get_id(agent) for agent in self.agents]
        node_obs_n_all, adj_n_all = self._get_graph_obs()

        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents

        return obs_n, agent_id_n, node_obs_n, adj_n

    def step(self, action_n: List) -> Tuple[List[arr], List[arr], List[arr], List[arr], List[arr], List[bool], List[Dict]]:
        self.current_step_in_episode += 1
        self.episode_steps_count +=1 # 记录实际运行的步数

        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        for agent in self.agents:
            if self.discrete_action:
                 agent.vel = agent.action * self.speed
            else:
                 agent.vel = agent.action * self.speed
            agent.pos = (agent.pos + agent.vel) % self.world_size

        payoffs = self._compute_payoffs()
        self._record_payoffs(payoffs) # agent.last_payoff 更新
        self._update_strategies(payoffs) # agent.strategy 更新

        # --- 累积当前步的奖励和合作状态 ---
        current_step_rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            current_step_rewards[i] = agent.last_payoff[0] # 获取刚记录的收益作为本步奖励
            self.episode_total_rewards[i] += current_step_rewards[i]
            if agent.strategy[0] == 1: # 如果合作
                self.episode_cooperation_counts[i] += 1
        # ----------------------------------

        if self.cache_dists:
            self.calculate_distances()
            self.update_graph()

        obs_n = [self._get_obs(agent) for agent in self.agents]
        agent_id_n = [self._get_id(agent) for agent in self.agents]
        node_obs_n_all, adj_n_all = self._get_graph_obs()
        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents

        reward_n_for_buffer = [] # 用于存入 Buffer 的奖励列表 (每个元素是 shape (1,) 的数组)
        done_n = []
        info_n = [] # 用于存储每个 agent 的 info 字典

        # 判断是否达到环境的 max_cycles
        is_episode_done = (self.current_step_in_episode >= self.max_cycles)

        for i, agent in enumerate(self.agents):
            reward_n_for_buffer.append(np.array([current_step_rewards[i]], dtype=np.float32)) # Buffer 需要 (1,) 形状
            done_n.append(is_episode_done) # 所有 agent 同时 done

            agent_info = {
                "individual_reward_step": current_step_rewards[i], # 当前步骤的个体奖励
                "strategy_step": agent.strategy[0], # 当前步骤的策略
            }
            # --- 如果 episode 结束，添加聚合指标到 info ---
            if is_episode_done:
                # 1. 总累积奖励 (Social GDP for this agent's perspective, or for the whole system)
                #    这里我们放入整个系统的总累积奖励
                agent_info["episode_social_gdp"] = np.sum(self.episode_total_rewards)

                # 2. 平均每步每智能体奖励
                if self.episode_steps_count > 0:
                    agent_info["episode_avg_reward_per_agent_step"] = np.sum(self.episode_total_rewards) / (self.num_agents * self.episode_steps_count)
                else:
                    agent_info["episode_avg_reward_per_agent_step"] = 0.0

                # 3. 平均合作率
                #    计算的是所有智能体在整个 episode 中的平均合作次数比例
                if self.episode_steps_count > 0:
                     agent_info["episode_avg_cooperation_rate"] = np.sum(self.episode_cooperation_counts) / (self.num_agents * self.episode_steps_count)
                else:
                     agent_info["episode_avg_cooperation_rate"] = 0.0

                # 也可以添加个体累积奖励
                agent_info["episode_individual_cumulative_reward"] = self.episode_total_rewards[i]

            info_n.append(agent_info)
            # ----------------------------------------------

        # 如果 episode 结束，重置 episode 内部步数计数器，为下一个 VecEnv 的 reset 做准备
        # （虽然 DummyVecEnv 会在 done 后自动 reset，但 SubprocVecEnv 的 worker 也会）
        if is_episode_done:
            self.current_step_in_episode = 0 # 重置，这样如果 Runner 的 episode_length > max_cycles，也能正确处理
            # 累加器在下一次环境 reset 时重置

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n_for_buffer, done_n, info_n

    # --- 核心博弈逻辑 (calculate_distances, _compute_payoffs, _update_strategies, _record_payoffs) ---
    # --- 这些方法保持不变 ---
    def calculate_distances(self):
        num_entities = len(self.agents)
        positions = np.array([agent.pos for agent in self.agents])
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2
        self.cached_dist_vect = delta
        self.cached_dist_mag = np.linalg.norm(delta, axis=2)

    def _compute_payoffs(self) -> Dict[int, float]:
        N = self.num_agents
        payoffs = {i: 0.0 for i in range(N)}
        dmat = self.cached_dist_mag
        for i in range(N):
            dists = dmat[i]
            group_indices = np.where(dists <= self.radius)[0]
            if len(group_indices) == 0: continue
            contrib = sum(int(self.agents[j].strategy[0]) * self.cost for j in group_indices)
            pool = contrib * self.r
            share = pool / len(group_indices) if len(group_indices) > 0 else 0.0
            for j in group_indices:
                cost_paid = self.cost if self.agents[j].strategy[0] == 1 else 0.0
                payoffs[j] += share - cost_paid
        return payoffs

    def _update_strategies(self, payoffs: Dict[int, float]):
        N = self.num_agents
        dmat = self.cached_dist_mag
        next_strategies = [agent.strategy.copy() for agent in self.agents]
        for i in range(N):
            dists = dmat[i]
            neighbor_indices = np.where((dists > 0) & (dists <= self.radius))[0]
            if len(neighbor_indices) == 0: continue
            j = np.random.choice(neighbor_indices)
            delta_payoff = payoffs[j] - payoffs[i]
            prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
            if np.random.rand() < prob_adopt:
                next_strategies[i] = self.agents[j].strategy.copy()
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]

    def _record_payoffs(self, payoffs: Dict[int, float]):
        for i, ai in enumerate(self.agents):
            ai.last_payoff = np.array([payoffs[i]], dtype=np.float32)

    # --- 图更新逻辑 (不变) ---
    def update_graph(self):
        if self.cached_dist_mag is None: self.calculate_distances()
        dists = self.cached_dist_mag
        connect_mask = (dists > 0) & (dists <= self.radius)
        row, col = np.where(connect_mask)
        self.edge_list = np.stack([row, col]) # 已经是 (2, E)
        self.edge_weight = dists[row, col]

    # --- 观测、ID、图观测、动作设置 (主要修改 _get_agent_feat) ---
    def _get_agent_feat(self, agent: Agent) -> arr:
        """
        构造指定智能体的节点/局部特征向量 (float32)。
        现在不包含实体类型 ID。
        """
        pos = agent.pos / self.world_size
        vel = np.clip(agent.vel / (self.speed + 1e-8), -1.0, 1.0)
        strategy = agent.strategy
        last_payoff = agent.last_payoff

        features = np.hstack([
            pos.flatten(),          # (2,)
            vel.flatten(),          # (2,)
            strategy.flatten(),     # (1,)
            last_payoff.flatten(),  # (1,)
        ]).astype(np.float32)       # 总维度现在是 6

        if np.isnan(features).any() or np.isinf(features).any():
            print(f"警告: agent {agent.id} 的特征中发现 NaN/Inf！已替换为 0。")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def _get_obs(self, agent: Agent) -> arr:
        return self._get_agent_feat(agent)

    def _get_reward(self, agent: Agent) -> np.ndarray:
        # 这个函数现在不直接被 Runner 使用，奖励在 step 中处理并放入 reward_n_for_buffer
        # 但可以保留，如果其他地方需要单步奖励
        return np.array([float(agent.last_payoff[0])], dtype=np.float32)

    def _get_done(self, agent: Agent) -> bool: # agent 参数保留，但所有 agent 同时 done
        # 这个函数现在也不直接被 Runner 使用，done 在 step 中统一处理
        return self.current_step_in_episode >= self.max_cycles

    def _get_info(self, agent: Agent) -> Dict:
        # 这个函数现在在 step 中被调用来构建 agent_info
        # 返回的是单步信息，episode 聚合信息在 step 中添加到 agent_info
        return {
            "individual_reward_step": float(agent.last_payoff[0]),
            "strategy_step": int(agent.strategy[0]),
        }

    def _get_id(self, agent: Agent) -> arr:
        return np.array([agent.id], dtype=np.int32)

    def _get_graph_obs(self) -> Tuple[arr, arr]:
        node_features = [self._get_agent_feat(agent) for agent in self.agents]
        node_obs = np.array(node_features, dtype=np.float32)
        adj = self.cached_dist_mag.astype(np.float32) if self.cached_dist_mag is not None else \
              np.zeros((self.num_agents, self.num_agents), dtype=np.float32) # 处理未初始化情况
        return node_obs, adj

    def _set_action(self, action, agent: Agent, action_space) -> None:
        # _set_action 逻辑保持不变 (基于上次修正后的版本)
        agent.action = np.zeros(2, dtype=np.float32)
        if isinstance(action, (list, tuple)): current_action = np.array(action)
        else: current_action = action

        if self.discrete_action:
            action_idx = 0
            if isinstance(current_action, np.ndarray):
                if current_action.size == 1: action_idx = int(current_action.item())
                else: action_idx = np.argmax(current_action)
            else:
                try: action_idx = int(current_action)
                except (TypeError, ValueError): action_idx = 0
            direction = {0:(0.0,0.0), 1:(-1.0,0.0), 2:(1.0,0.0), 3:(0.0,-1.0), 4:(0.0,1.0)}
            if action_idx in direction:
                dx, dy = direction[action_idx]
                agent.action[0], agent.action[1] = dx, dy
        else: # 连续动作
            try:
                cont_action = np.array(current_action, dtype=np.float32).flatten()
                if cont_action.shape[0] == 2:
                    agent.action = np.clip(cont_action, -1.0, 1.0) # 假设动作是归一化的速度比例
                elif cont_action.shape[0] == 1: # 如果只收到一个值，默认控制x轴
                    agent.action[0] = np.clip(cont_action[0], -1.0, 1.0)
            except (TypeError, ValueError): pass # 保持 agent.action 为 [0,0]

# --- END OF FILE marl_env.py ---