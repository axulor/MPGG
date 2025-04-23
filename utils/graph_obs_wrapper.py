# 导入所需的库
from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper # PettingZoo 提供的包装器基类
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
import gymnasium

from envs.migratory_pgg_env import MigratoryPGGEnv


# 定义图观测包装器
class GraphObservationWrapper(BaseParallelWrapper):
    """
    将 MigratoryPGGEnv 的观测转换为适合 GNN 处理的图结构。

    为每个智能体提供以下观测信息:
    - obs: 当前智能体自身的特征 (strategy, last_payoff, pos_x, pos_y)。
    - share_obs: 所有智能体的位置信息拼接成的全局向量。
    - node_obs: 所有智能体的特征矩阵 (N, num_features)。
    - adj: 邻接/距离矩阵 (N, N)，只记录半径内的距离。
    """
    def __init__(self, env: MigratoryPGGEnv):
        """
        初始化包装器。

        Args:
            env: 要包装的 MigratoryPGGEnv 环境实例。
        """
        super().__init__(env)
        self.env = env # 显式存储原始环境

        # 检查原始环境是否已初始化（获取 N 等参数）
        if not hasattr(self.env, 'N'):
            # 如果环境未重置，可能无法获取 N，这里先尝试访问
            # 或者要求传入的 env 必须是已 reset 过至少一次的
            # 为了健壮性，可以先调用一次 reset
            print("Warning: Env might not be initialized. Calling reset() to ensure parameters are available.")
            self.env.reset()


        self._agent_list = list(self.env.possible_agents) # 保证智能体顺序固定
        self._agent_name_to_id = {name: i for i, name in enumerate(self._agent_list)}
        self.share_observation_space = [] # 为兼容 MAPPO 代码添加

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.n)
        ]

        self._define_graph_spaces()

    def _get_toroidal_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算考虑周期边界的两个位置之间的欧氏距离"""
        delta = np.abs(pos1 - pos2)
        delta = np.minimum(delta, self.env.size - delta)
        return np.linalg.norm(delta)

    def _define_graph_spaces(self):
        """定义新的图结构观测空间"""
        N = self.env.N
        # 单个智能体的特征: strategy(1) + last_payoff(1) + pos_x(1) + pos_y(1) = 4
        # 注意：我们把位置信息也加入到 obs 和 node_obs 中，这通常对 GNN 有用
        agent_feature_dim = 4

        # 1. obs_space: 单个智能体特征
        obs_feature_space = spaces.Box(low=-np.inf, high=np.inf, shape=(agent_feature_dim,), dtype=np.float32)

        # 2. share_obs_space: 全局位置信息 (N * 2)
        share_obs_space = spaces.Box(low=0.0, high=self.env.size, shape=(N * 2,), dtype=np.float32)

        # 3. node_obs_space: 节点特征矩阵 (N, agent_feature_dim)
        node_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N, agent_feature_dim), dtype=np.float32)

        # 4. edge_obs_space: 邻接/距离矩阵 (N, N)
        #    值为 0 表示无连接或自身，> 0 表示半径内的距离
        adj_space = spaces.Box(low=0.0, high=self.env.radius, shape=(N, N), dtype=np.float32)

        # 更新每个智能体的观测空间
        self.observation_spaces = {
            agent: spaces.Dict({
                "obs": obs_feature_space,
                "share_obs": share_obs_space,
                "node_obs": node_obs_space,
                "adj": adj_space,
                # "agent_id": spaces.Discrete(N)    # agent_id 不是环境状态的一部分，通常不放在space里
                                                    # 它是在获取观测时，与观测一起给到智能体策略的元信息
            }) for agent in self.env.possible_agents
        }

    def _compute_graph_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """计算所有智能体的图结构观测"""
        N = self.env.N
        agent_feature_dim = 4 # strategy, payoff, pos_x, pos_y

        node_obs = np.zeros((N, agent_feature_dim), dtype=np.float32)
        share_obs_list = []
        adj = np.zeros((N, N), dtype=np.float32)
        current_pos = {} # 临时存储位置，避免重复访问字典

        # 1. 构建节点特征矩阵 (node_obs) 和部分共享观测 (share_obs_list)
        for i, agent_name in enumerate(self._agent_list):
            if agent_name in self.env.pos: # 确保智能体还存在
                pos = self.env.pos[agent_name]
                strategy = float(self.env.strategy.get(agent_name, 0)) # 使用 get 以防万一
                last_payoff = float(self.env.last_payoff.get(agent_name, 0.0))

                node_obs[i, 0] = strategy
                node_obs[i, 1] = last_payoff
                node_obs[i, 2] = pos[0]
                node_obs[i, 3] = pos[1]

                share_obs_list.extend(pos)
                current_pos[agent_name] = pos
            else:
                # 如果智能体已终止，用默认值填充？或根据情况处理
                # 这里假设我们总是为 possible_agents 中的所有智能体计算观测
                # 如果智能体已终止，其特征可能无意义，但结构需要保持
                # GNN 处理时可能需要 mask 掉无效节点
                pass # 保持为 0

        share_obs = np.array(share_obs_list, dtype=np.float32)

        # 2. 构建邻接/距离矩阵 (adj)
        agent_indices = list(range(N))
        for i in agent_indices:
            agent_name_i = self._agent_list[i]
            if agent_name_i not in current_pos: continue # 跳过已移除的智能体

            pos_i = current_pos[agent_name_i]
            for j in range(i + 1, N): # 避免重复计算和自环
                agent_name_j = self._agent_list[j]
                if agent_name_j not in current_pos: continue # 跳过已移除的智能体

                pos_j = current_pos[agent_name_j]
                dist = self._get_toroidal_distance(pos_i, pos_j)

                if dist <= self.env.radius:
                    adj[i, j] = dist
                    adj[j, i] = dist # 矩阵是对称的

        # 3. 组合每个智能体的观测
        graph_observations = {}
        for agent_name in self.env.agents: # 只为当前活跃的智能体生成观测
            agent_id = self._agent_name_to_id[agent_name]
            agent_obs_features = node_obs[agent_id]

            graph_observations[agent_name] = {
                "obs": agent_obs_features,
                "share_obs": share_obs.copy(), # 复制以防万一
                "node_obs": node_obs.copy(),   # 复制以防万一
                "adj": adj.copy()              # 复制以防万一
                # agent_id 可以由调用者通过 agent_name 查询 _agent_name_to_id 获得
            }

        return graph_observations

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, dict]]:
        """重置环境并返回图结构的初始观测"""
        # 注意：PettingZoo 1.24+ reset 返回 obs, infos
        _, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents # 同步活跃智能体列表
        observations = self._compute_graph_observations()
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """执行一步环境，并返回图结构的观测"""
        # step 返回 obs, rewards, terminations, truncations, infos
        _, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents # 同步活跃智能体列表
        observations = self._compute_graph_observations()
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        """返回指定智能体的观测空间"""
        return self.observation_spaces[agent]

    # Wrapper 会自动代理未被覆盖的方法和属性到 self.env
    # 例如 render, close, state 等方法可以直接调用 wrapper.render()
    # 如果需要修改 state() 的行为，也可以覆盖它

    # 我们可以添加一个方法来方便地获取 agent_id
    def get_agent_id(self, agent_name: str) -> int:
        """获取智能体名称对应的索引 ID"""
        return self._agent_name_to_id[agent_name]

    @property
    def possible_agents(self) -> List[str]:
        """覆盖以确保返回正确的列表"""
        return self.env.possible_agents

    # 如果需要访问原始环境的 state()，可以通过 self.env.state()
    # 如果想让包装器也有 state() 并返回类似结构，可以添加：
    # def state(self) -> np.ndarray:
    #     """ 返回原始环境的全局状态 """
    #     if hasattr(self.env, 'state'):
    #         return self.env.state()
    #     else:
    #         # 或者根据 node_obs 重建一个 state 表示
    #         # 注意原始 state 包含速度，而我们包装的 node_obs 默认不包含
    #         raise NotImplementedError("Wrapped environment does not have a state() method or wrapper needs to implement its own.")
    #         # return self._compute_graph_observations()[self.agents[0]]["node_obs"].flatten() # 示例，不完全等同

