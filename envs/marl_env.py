# --- START OF FILE marl_env.py ---

import gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import argparse
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# == Agent Class Definition ==
# ==============================================================================

class Agent:
    """
    迁徙公共物品博弈环境中的智能体类，封装单个智能体的状态信息。
    此类现在是 MultiAgentGraphEnv 的一部分或在其内部使用。

    属性:
        id (int) : 智能体唯一标识符
        name (str): 智能体名称
        pos (np.ndarray): 智能体在二维空间中的位置向量，shape=(2,)
        vel (np.ndarray): 智能体的物理速度向量，shape=(2,)，包含方向和大小
        strategy (np.ndarray): 当前策略，一维数组，shape=(1,), 0=背叛(defect)、1=合作(cooperate)
        last_payoff (np.ndarray): 上一次公共物品博弈的收益，一维数组，shape=(1,)
        action (np.ndarray): 智能体当前的动作意图 (通常由 _set_action 设置)
    """
    def __init__(self):
        self.id = None
        self.name = None
        # 初始化位置和速度，实际在 reset 时做随机初始化
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        # 初始策略为 0，存为一维数组
        self.strategy = np.array([0], dtype=np.int32)  # shape: (1,)
        # 上一次收益，存为一维数组
        self.last_payoff = np.array([0.0], dtype=np.float32)  # shape: (1,)
        # 智能体动作 (物理上的)
        self.action = np.zeros(2, dtype=np.float32) # 例如: [dx, dy]

# ==============================================================================
# == Merged MultiAgentGraphEnv Class Definition ==
# ==============================================================================

class MultiAgentGraphEnv(gym.Env):
    """
    合并了 World, Scenario 和原 Gym Wrapper 功能的多智能体图环境类。

    此类负责:
    1. 初始化环境参数和智能体。
    2. 管理智能体状态 (位置, 速度, 策略, 收益)。
    3. 执行环境的核心模拟步骤 (移动, 博弈, 策略演化)。
    4. 计算和缓存智能体间的距离。
    5. 构建和更新图结构 (节点特征, 邻接信息, 边列表)。
    6. 提供标准的 Gym 环境接口 (step, reset, observation_space, action_space)。
    7. 计算观测、奖励、完成状态和信息
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, args: argparse.Namespace):
        """
        初始化环境。

        Args:
            args (argparse.Namespace): 包含环境配置参数的对象。
                需要包含: num_agents, world_size, speed, max_cycles, radius,
                        cost, r, beta, scenario_name (可选), discrete_action (可选)
        """
        super().__init__()

        # -- 1. 从 args 初始化环境和博弈参数 --
        self.num_agents = args.num_agents if hasattr(args, 'num_agents') else 20 # 默认值
        self.world_size = args.world_size if hasattr(args, 'world_size') else 100.0
        self.speed = args.speed if hasattr(args, 'speed') else 1.0
        self.max_cycles = args.max_cycles if hasattr(args, 'max_cycles') else 500
        self.radius = args.radius if hasattr(args, 'radius') else 10.0 # 用于博弈交互和图构建
        self.cost = args.cost if hasattr(args, 'cost') else 1.0
        self.r = args.r if hasattr(args, 'r') else 1.0 # 公共物品博弈乘数
        self.beta = args.beta if hasattr(args, 'beta') else 0.5 # Fermi 函数噪声参数
        self.scenario_name = args.scenario_name if hasattr(args, 'scenario_name') else "mpgg_graph" # 场景名称
        self.discrete_action = args.discrete_action if hasattr(args, 'discrete_action') else True # 默认离散动作

        # -- 2. 初始化智能体列表 --
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # -- 3. 初始化环境状态 --
        self.current_step = 0 # Gym 环境的步数计数器 (从 reset 开始)
        self.current_time = 0 # 世界内部的时间步 (可能与 Gym step 不同，但这里保持一致)

        # -- 4. 图和距离相关属性 --
        self.edge_list = None       # 边列表 (2, E)
        self.edge_weight = None     # 边权重 (E,)
        self.cached_dist_vect = None # 缓存距离向量 (N, N, 2)
        self.cached_dist_mag = None  # 缓存距离大小 (N, N)
        self.cache_dists = True     # 始终缓存距离，因为图观测需要

        # -- 5. 配置 Gym 空间 --
        # 动作空间 (所有智能体相同)
        if self.discrete_action:
            # 离散动作: 0:不动, 1:左, 2:右, 3:下, 4:上
            self.agent_action_space = spaces.Discrete(5)
        else:
            # 连续动作: [dx, dy] in [-1, 1]
            self.agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = [self.agent_action_space] * self.num_agents

        # 观测空间 (需要基于 _get_agent_feat 确定维度)
        # 先创建一个临时 agent 来获取特征维度
        _temp_agent = Agent()
        _temp_agent.pos = np.zeros(2)
        _temp_agent.vel = np.zeros(2)
        _temp_agent.strategy = np.array([0])
        _temp_agent.last_payoff = np.array([0.0])
        _obs_sample = self._get_agent_feat(_temp_agent)
        obs_dim = _obs_sample.shape[0]

        self.agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = [self.agent_observation_space] * self.num_agents

        # 共享观测空间 (所有智能体观测的拼接)
        share_obs_dim = obs_dim * self.num_agents
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
        ] * self.num_agents # GymMA padrão espera uma lista

        # 图相关观测空间
        # 节点特征维度与 agent 观测维度相同
        node_obs_dim = (self.num_agents, obs_dim)
        adj_dim = (self.num_agents, self.num_agents)
        agent_id_dim = (1,) #  ID 是标量
        edge_dim = (1,) # 边特征空间维度，定为1

        self.node_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=node_obs_dim, dtype=np.float32)] * self.num_agents
        self.adj_observation_space = [spaces.Box(low=0, high=+np.inf, shape=adj_dim, dtype=np.float32)] * self.num_agents # 邻接矩阵是距离，非负
        self.agent_id_observation_space = [spaces.Box(low=0, high=self.num_agents-1, shape=agent_id_dim, dtype=np.int32)] * self.num_agents # ID 是整数索引
        self.share_agent_id_observation_space = [
            spaces.Box(
                low=0, high=self.num_agents-1,
                 shape=(self.num_agents * agent_id_dim[0],), # 所有 ID 拼接
                dtype=np.int32,
            )
        ] * self.num_agents

        # 边特征空间
        self.edge_observation_space = [(spaces.Box(low=-np.inf, high=+np.inf, shape = edge_dim, dtype=np.float32))] * self.num_agents


    def seed(self, seed=None):
        """设置环境的随机种子"""
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)


    def reset(self) -> Tuple[List[arr], List[arr], List[arr], List[arr]]:
        """
        重置环境到初始状态。
        相当于原 Scenario.reset_world 的逻辑。

        Returns:
            A tuple containing the initial observations for each agent:
            - obs_n: 每个智能体的初始局部观察列表，列表元素shape=(7,)
            - agent_id_n: 每个智能体的初始ID列表
            - node_obs_n: 每个智能体的初始节点特征观察列表, 列表元素shape=(100, 7)
            - adj_n: 每个智能体的初始邻接矩阵观察列表
        """
        self.current_step = 0
        self.current_time = 0

        # --- 重置智能体状态 ---
        shuffled_agents = list(self.agents) # 创建副本以进行 shuffle
        np.random.shuffle(shuffled_agents)
        half = (self.num_agents + 1) // 2
        cooperators = set(shuffled_agents[:half]) # 取一半作为合作者

        for agent in self.agents:
            agent.pos = np.random.rand(2) * self.world_size   # 随机位置
            theta = np.random.rand() * 2 * np.pi
            agent.vel = self.speed * np.array([np.cos(theta), np.sin(theta)])  # 随机速度方向
            agent.last_payoff = np.array([0.0], dtype=np.float32) # 重置收益
            agent.strategy = np.array([1 if agent in cooperators else 0], dtype=np.int32) # 设置初始策略
            agent.action = np.zeros(2, dtype=np.float32) # 重置动作

        # --- 计算初始距离和图结构 ---
        self.calculate_distances()
        self.update_graph()

        # --- 获取初始观测 ---
        obs_n = [self._get_obs(agent) for agent in self.agents]
        agent_id_n = [self._get_id(agent) for agent in self.agents]
        node_obs_n_all, adj_n_all = self._get_graph_obs() # 获取全局图信息

        # 所有 agent 接收相同图结构信息
        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents

        return obs_n, agent_id_n, node_obs_n, adj_n


    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        """
        环境执行一个时间步。
        包含原 World.step 的逻辑和 Gym 环境的接口处理。

        Args:
            action_n: 包含每个智能体动作的列表。

        Returns:
            A tuple containing:
            - obs_n: 每个智能体的局部观察列表。
            - agent_id_n: 每个智能体的ID列表。
            - node_obs_n: 每个智能体的节点特征观察列表。
            - adj_n: 每个智能体的邻接矩阵观察列表。
            - reward_n: 每个智能体的奖励列表。
            - done_n: 每个智能体的完成状态列表。
            - info_n: 每个智能体的额外信息字典列表。
        """
        self.current_step += 1
        self.current_time += 1 # 更新内部时间

        # -- 1. 为每个智能体设置动作 --
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # -- 2. 更新智能体位置 (物理引擎) --
        for agent in self.agents:
            # 动作决定速度变化或直接是速度向量 (取决于场景设计，这里假设动作为速度)
            # 如果动作是施加力，则需要更新速度 agent.vel += agent.action
            # 如果动作是直接设置速度方向/大小
            if self.discrete_action:
                # agent.action 是方向向量 [dx, dy] 或 [0, 0]
                 agent.vel = agent.action * self.speed # 将方向动作转换为速度
            else:
                # 连续动作通常直接控制速度或力，这里假设控制速度
                 agent.vel = agent.action * self.speed # 假设 action 是 [-1,1] 的方向/比例

            # 应用速度更新位置
            agent.pos = (agent.pos + agent.vel) % self.world_size

        # -- 3. 计算博弈收益 --
        payoffs = self._compute_payoffs()

        # -- 4. 记录收益 --
        self._record_payoffs(payoffs)

        # -- 5. 策略演化 (Fermi Rule) --
        self._update_strategies(payoffs)

        # -- 6. 更新距离缓存和图结构 --
        if self.cache_dists:
            self.calculate_distances()
            self.update_graph() # 确保图结构基于新位置更新

        # -- 7. 获取每个智能体的返回信息 --
        obs_n, reward_n, done_n, info_n = [], [], [], []
        node_obs_n_all, adj_n_all = self._get_graph_obs() # 全局图信息
        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents
        agent_id_n = [self._get_id(agent) for agent in self.agents]

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))

        # 检查是否达到最大步数，如果是则所有智能体都完成
        if self.current_step >= self.max_cycles:
            done_n = [True] * self.num_agents

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

    # ==========================================================================
    # == Internal Helper Methods (Core Logic, previously in World) ==
    # ==========================================================================

    def calculate_distances(self):
        """
        计算并缓存所有智能体之间的周期边界下的距离向量和距离矩阵。
        """
        num_entities = len(self.agents)
        vect = np.zeros((num_entities, num_entities, 2), dtype=np.float32)
        mag = np.zeros((num_entities, num_entities), dtype=np.float32)
        positions = np.array([agent.pos for agent in self.agents]) # 获取所有位置

        # 使用广播高效计算所有对之间的差异
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :] # Shape (N, N, 2)

        # 应用周期性边界条件
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2

        # 计算距离大小
        mag = np.linalg.norm(delta, axis=2) # Shape (N, N)

        # 更新缓存
        self.cached_dist_vect = delta
        self.cached_dist_mag = mag


    def _compute_payoffs(self) -> Dict[int, float]:
        """
        计算并返回所有智能体的MPGG博弈收益。
        使用缓存的距离矩阵 self.cached_dist_mag。
        """
        N = self.num_agents
        payoffs = {i: 0.0 for i in range(N)}
        dmat = self.cached_dist_mag # 使用缓存的距离

        for i, ai in enumerate(self.agents):
            # 找到邻居 (包括自己)
            dists = dmat[i]
            # 找出距离在 [0, radius] 内的智能体索引 (包括自己, dists[i] == 0)
            group_indices = np.where(dists <= self.radius)[0]

            if len(group_indices) == 0: # 不太可能，因为自己总在里面
                continue

            # 计算小组贡献总和
            contrib = sum(int(self.agents[j].strategy[0]) * self.cost for j in group_indices)
            pool = contrib * self.r # 乘以增益因子
            share = pool / len(group_indices) # 平均分配

            # 为小组内每个成员增加收益份额，并减去其贡献成本（如果合作）
            for j in group_indices:
                cost_paid = self.cost if self.agents[j].strategy[0] == 1 else 0
                payoffs[j] += share - cost_paid

        return payoffs


    def _update_strategies(self, payoffs: Dict[int, float]):
        """
        使用 Fermi 函数基于邻居收益差进行策略更新。
        """
        N = self.num_agents
        dmat = self.cached_dist_mag
        next_strategies = [agent.strategy.copy() for agent in self.agents] # 存储下一步策略，避免更新过程中的干扰

        for i, ai in enumerate(self.agents):
            dists = dmat[i]
            # 找到严格在半径内的邻居 (不包括自己)
            neighbor_indices = np.where((dists > 0) & (dists <= self.radius))[0]

            if len(neighbor_indices) == 0:
                continue # 没有邻居可供学习

            # 随机选择一个邻居 j
            j = np.random.choice(neighbor_indices)

            # 计算收益差和采纳概率 (Fermi rule)
            delta_payoff = payoffs[j] - payoffs[i]
            prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))

            # 以概率 prob_adopt 采纳邻居 j 的策略
            if np.random.rand() < prob_adopt:
                next_strategies[i] = self.agents[j].strategy.copy()

        # 应用更新后的策略
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]


    def _record_payoffs(self, payoffs: Dict[int, float]):
        """
        将计算出的 payoffs 存入每个智能体的 last_payoff 属性。
        """
        for i, ai in enumerate(self.agents):
            ai.last_payoff = np.array([payoffs[i]], dtype=np.float32)


    def _compute_distances_row(self, idx: int) -> np.ndarray:
        """
        (未使用，因为 calculate_distances 更高效)
        计算单个智能体到所有其他智能体的周期距离。
        """
        row = []
        pi = self.agents[idx].pos
        for aj in self.agents:
            delta = (pi - aj.pos + self.world_size/2) % self.world_size - self.world_size/2
            row.append(np.linalg.norm(delta))
        return np.array(row, dtype=np.float32)


    # ==========================================================================
    # == Internal Helper Methods (Scenario Logic) ==
    # ==========================================================================

    def update_graph(self):
        """
        构建并更新环境中的图结构 (边列表和边权重)。
        使用缓存的距离矩阵 self.cached_dist_mag。
        """
        dists = self.cached_dist_mag
        # 创建连接矩阵: True 如果 0 < dist <= radius
        connect_mask = (dists > 0) & (dists <= self.radius)

        # 获取连接的行、列索引 (边的起点和终点)
        row, col = np.where(connect_mask)

        # 构建边列表 (2, E)
        self.edge_list = np.stack([row, col])
        # 获取对应边的权重 (距离)
        self.edge_weight = dists[row, col]


    def _get_agent_feat(self, agent: Agent) -> arr:
        """
        构造指定智能体的节点/局部特征向量 (float32)
        包括归一化位置、归一化速度、策略、上一轮收益，以及实体类型ID
        """
        pos = agent.pos / self.world_size
        vel = np.clip(agent.vel / (self.speed + 1e-8), -1.0, 1.0) # TODO 似乎有点问题
        strategy = agent.strategy
        last_payoff = agent.last_payoff
        
        # --- 兼容性设置 ---
        entity_type_id = 0 # 0 代表 Agent 类型
        # -----------

        features = np.hstack([
            pos.flatten(),              # 位置 (2,)
            vel.flatten(),              # 速度 (2,)
            strategy.flatten(),         # 策略(1,)
            last_payoff.flatten(),      # 收益(1,)
            np.array([entity_type_id], dtype=np.float32) # 类型ID(1,) 
        ]).astype(np.float32)       # 总维度变为 7

        # TODO 返回值检查
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"WARNING: NaN or Inf found in features for agent {agent.id}! Replacing with 0.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features


    def _get_obs(self, agent: Agent) -> arr:
        """获取指定智能体的局部观测 (即其自身特征)。"""
        return self._get_agent_feat(agent)


    def _get_reward(self, agent: Agent) -> np.ndarray: # Change return type hint
        """获取指定智能体的奖励 (以 shape (1,) 的数组形式返回)。"""
        return np.array([float(agent.last_payoff[0])], dtype=np.float32)


    def _get_done(self, agent: Agent) -> bool:
        """获取指定智能体的完成状态 (基于最大步数)。"""
        # 在这个场景中，所有智能体同时完成
        return self.current_step >= self.max_cycles


    def _get_info(self, agent: Agent) -> Dict:
        """获取指定智能体的额外信息。"""
        # 可以包含个体奖励，策略等
        return {
            "individual_reward": float(agent.last_payoff[0]),
            "strategy": int(agent.strategy[0])
        }


    def _get_id(self, agent: Agent) -> arr:
        """获取指定智能体的 ID。"""
        return np.array([agent.id], dtype=np.int32)


    def _get_graph_obs(self) -> Tuple[arr, arr]:
        """
        构建并返回全局图结构观测：节点特征矩阵和邻接距离矩阵
        注意：此实现返回所有智能体共享的全局信息

        Returns:
            - node_obs (np.ndarray): 所有节点的特征矩阵 (N, num_node_feats)。
            - adj (np.ndarray): 邻接距离矩阵 (N, N)。
        """
        # 1. 获取所有节点的特征
        node_features = [self._get_agent_feat(agent) for agent in self.agents]
        node_obs = np.array(node_features, dtype=np.float32)

        # 2. 获取缓存的距离矩阵作为邻接信息
        adj = self.cached_dist_mag.astype(np.float32)

        return node_obs, adj


    # ==========================================================================
    # == Internal Helper Methods (Action Setting) ==
    # ==========================================================================

        # marl_env.py

    def _set_action(self, action, agent: Agent, action_space) -> None:
        """
        为指定智能体设置动作。
        将 Gym 动作空间的输出转换为智能体内部的物理动作 `agent.action`。
        """
        # 清零之前的动作意图
        agent.action = np.zeros(2, dtype=np.float32)

        # 确保 action 是 NumPy 数组或标量以便处理
        if isinstance(action, (list, tuple)):
            current_action = np.array(action)
        else:
            current_action = action # 可能已经是标量或 NumPy 数组

        if self.discrete_action:
            # --- 修改开始 ---
            if isinstance(current_action, np.ndarray):
                # 如果输入是 NumPy 数组
                if current_action.size == 1:
                    # 如果是只有一个元素的数组 (例如 np.array([2]))，取出其中的值
                    action_idx = int(current_action.item())
                else:
                    # 如果是多元素的数组 (概率分布或 one-hot)，取最大值的索引
                    action_idx = np.argmax(current_action)
                    # 添加一个打印语句来确认，如果需要的话
                    # print(f"DEBUG: Agent {agent.id} received array action: {current_action}, using argmax index: {action_idx}")
            else:
                # 如果输入不是数组，假定它已经是标量索引
                try:
                    action_idx = int(current_action)
                except (TypeError, ValueError) as e:
                    # 添加错误处理，以防万一收到意外的非数字类型
                    print(f"ERROR: Agent {agent.id} received unexpected action format for discrete action: {current_action} (type: {type(current_action)}). Error: {e}")
                    action_idx = 0 # 设置一个默认动作，例如“不动”
            # --- 修改结束 ---

            # 方向映射表 (保持不变)
            direction = {
                0: (0.0, 0.0),   # 不动
                1: (-1.0, 0.0),  # 左
                2: (1.0, 0.0),   # 右
                3: (0.0, -1.0),  # 下
                4: (0.0, 1.0),   # 上
            }
            if action_idx in direction:
                dx, dy = direction[action_idx]
                agent.action[0] = dx
                agent.action[1] = dy
            # 如果 action_idx 无效 (比如argmax返回了超出范围的索引，虽然不太可能)，agent.action 保持为 [0, 0]

        else:
            # 处理连续动作 (保持不变)
            # 确保 current_action 是扁平化的 NumPy 数组
            agent.action = np.array(current_action, dtype=np.float32).flatten()
            # 可能需要根据 action_space 的 shape 进行调整或截断
            if agent.action.shape[0] > 2:
                agent.action = agent.action[:2]
            elif agent.action.shape[0] < 2:
                # 如果维度不足，可能需要填充或者报错
                temp_action = np.zeros(2, dtype=np.float32)
                temp_action[:agent.action.shape[0]] = agent.action
                agent.action = temp_action


        # 注意：agent.action 现在存储的是意图（例如方向），
        # 在 step 的物理更新部分会用它结合 self.speed 来设置 agent.vel


    # ==========================================================================
    # == Rendering Methods (Optional) ==
    # ==========================================================================
    # def render(self, mode='human'):
    #     """(可选) 实现环境的可视化渲染"""
    #     pass

    # def close(self):
    #     """(可选) 清理渲染资源等"""
    #     pass

# --- END OF FILE marl_env.py ---