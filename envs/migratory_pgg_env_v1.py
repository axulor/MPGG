from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from agents.agent import Agent
import random
from typing import Dict, Tuple, Optional
from envs.visualizer import PygameVisualizer
import os
from datetime import datetime

class MigratoryPGGEnv(ParallelEnv):
    """多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game)"""

    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}

    def __init__(self,
                 L: int = 10,
                 l: float = 1.25,
                 N: int = 150,
                 preset_c: float = 1.0,
                 preset_r: float = 4.0,
                 custom_casinos: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None):
        """
        :param L: 总区域边长（单位任意，影响网格数量）
        :param l: 每个赌场边长（决定网格个数）
        :param N: 智能体数量
        :param preset_c: 默认成本
        :param preset_r: 默认收益倍率
        :param custom_casinos: 自定义赌场参数字典，键为 (row, col) 位置，值为 (c, r) 参数
        """
        super().__init__()
        # 基本参数
        self.grid_size = L                     # 总区域边长
        self.casino_size = l                   # 每个赌场边长
        self.N = N                             # 智能体数量
        self.phase = "pre_game"                # 当前阶段：pre_game 或 post_game
        self.num_cells = int(self.grid_size // self.casino_size)  # 网格数量（行/列）
        self.step_size = 1.0 / self.num_cells   # 用于移动时计算新位置

        self.agent_names = ["agent_" + str(i) for i in range(N)]
        self.agents = {}                       # 存储智能体实例
        self.timestep = 0                      # 时间步计数
        self.run_dir = None

        # 默认赌场参数
        self.preset_c = preset_c
        self.preset_r = preset_r
        self.custom_casinos = custom_casinos if custom_casinos is not None else {}

        # 定义动作空间
        # 博弈阶段：0 表示不贡献，1 表示贡献
        self.game_action_spaces = {agent: spaces.Discrete(2) for agent in self.agent_names}
        # 移动阶段：0: 上, 1: 下, 2: 左, 3: 右, 4: 不动
        self.move_action_spaces = {agent: spaces.Discrete(5) for agent in self.agent_names}

        # 扩展后的观测空间：状态为赌场位置 (row, col) 和赌场内个体数 n
        # 其中 n 的取值范围 0 到 N，所以采用 Discrete(N+1)
        self.observation_spaces = {
            agent: spaces.Tuple((
                spaces.Discrete(self.num_cells),  # row
                spaces.Discrete(self.num_cells),  # col
                spaces.Discrete(self.N + 1)         # n：赌场内个体数
            )) for agent in self.agent_names
        }
        self.action_spaces = self.game_action_spaces  # 初始阶段采用博弈动作空间

        self.casinos = []  # 最终的赌场列表，每个赌场为：(location, c, r)
        self.make_world()

    def make_world(self):
        """创建赌场与智能体"""
        self.create_casinos()
        self.create_agents()

    def create_casinos(self):
        """
        创建赌场。
        每个赌场为一个三元组：(location, c, r)
          - location: (row, col)，行列取值范围 0 ~ num_cells-1
          - c: 成本，若该位置在 custom_casinos 中，则使用自定义值，否则使用 preset_c
          - r: 收益倍率，同上
        """
        for row in range(self.num_cells):
            for col in range(self.num_cells):
                location = (row, col)
                if location in self.custom_casinos:
                    c, r = self.custom_casinos[location]
                else:
                    c, r = self.preset_c, self.preset_r
                self.casinos.append((location, c, r))

    def create_agents(self):
        """实例化智能体，并均匀分配到赌场"""
        self.agents = {}
        total_casinos = len(self.casinos)
        agent_id = 0
        # 均匀分配
        for casino in self.casinos:
            for _ in range(self.N // total_casinos):
                agent_name = f"agent_{agent_id}"
                self.agents[agent_name] = Agent(agent_name)
                self.agents[agent_name].set_current_casino(casino)
                agent_id += 1
        # 处理余数：随机分配
        random_casinos = random.sample(self.casinos, self.N % total_casinos)
        for casino in random_casinos:
            agent_name = f"agent_{agent_id}"
            self.agents[agent_name] = Agent(agent_name)
            self.agents[agent_name].set_current_casino(casino)
            agent_id += 1

    def get_observation(self, agent_name):
        """
        返回单个智能体的观测状态：
         - (row, col) 表示赌场位置，
         - n 表示当前该赌场内的智能体数量
        """
        agent = self.agents[agent_name]
        location, _, _ = agent.current_casino
        n = self.get_agent_count(agent.current_casino)
        return (location[0], location[1], n)

    def get_observations(self):
        """返回所有智能体的观测字典"""
        return {agent_name: self.get_observation(agent_name) for agent_name in self.agents}

    def get_agent_count(self, casino):
        """获取指定赌场中智能体数量"""
        return sum(1 for agent in self.agents.values() if agent.current_casino == casino)

    def get_cooperator_count(self, casino):
        """获取指定赌场中选择合作的智能体数量"""
        return sum(1 for agent in self.agents.values() if agent.current_casino == casino and agent.is_cooperator)

    def reset_obs(self, seed=None):
        """重置环境观测，返回所有智能体状态 (row, col, n)"""
        self.phase = "pre_game"
        self.timestep = 0
        observations = self.get_observations()
        self.action_spaces = self.game_action_spaces
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def reset(self, seed=None):
        """
        重置整个环境：
         - 重置 phase 和 timestep，
         - 重新分配智能体到赌场（可选：也可以保留智能体当前状态，只重置 phase），
         - 返回所有智能体的初始观测状态 (row, col, n) 以及 infos。
        """
        self.phase = "pre_game"
        self.timestep = 0
        #每个 episode 环境都重新随机分配智能体位置，调用 create_agents()
        self.create_agents()
        observations = self.get_observations()
        self.action_spaces = self.game_action_spaces
        infos = {agent: {} for agent in self.agents}
        return observations, infos


    def step(self, actions):
        """
        根据当前阶段执行决策：
         - 博弈阶段：每个智能体根据贡献决策获得奖励（Q-learning 状态基于 (row, col, n)），奖励计算使用赌场的 c 和 r；
         - 移动阶段：根据移动动作更新智能体所在赌场（即更新位置）。
        """
        rewards = {agent: 0 for agent in self.agents}
        if self.phase == "pre_game":
            # 博弈阶段
            for agent_name, game_action in actions.items():
                agent = self.agents[agent_name]
                agent.is_cooperator = (game_action == 1)
                rewards[agent_name] = self.get_reward(agent, game_action)
            self.phase = "post_game"
            observations = self.get_observations()
            self.action_spaces = self.move_action_spaces
        else:
            # 移动阶段
            for agent_name, move_action in actions.items():
                self.move_agents(agent_name, move_action)
            self.phase = "pre_game"
            observations = self.get_observations()
            self.action_spaces = self.game_action_spaces

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def get_reward(self, agent, game_action):
        """
        奖励计算：
          - 使用当前赌场的参数 c 和 r；
          - n 为该赌场中智能体总数，n_C 为合作智能体数量；
          - 若选择合作，奖励 = (r * n_C) / n - c；否则奖励 = (r * n_C) / n。
        """
        _, c, r = agent.current_casino
        n = self.get_agent_count(agent.current_casino)
        n_C = self.get_cooperator_count(agent.current_casino)
        if game_action == 1:
            reward = (r * n_C) / n - c
        else:
            reward = (r * n_C) / n
        return reward

    def move_agents(self, agent_name, move_action):
        """
        离散环状移动：
          - 当前状态为赌场位置 (row, col)；
          - 定义移动动作：0: 上 (-1,0), 1: 下 (+1,0), 2: 左 (0,-1), 3: 右 (0,+1), 4: 不动 (0,0)；
          - 使用模运算实现边界环绕；
          - 移动后，从 self.casinos 中查找对应位置的赌场，并更新智能体当前赌场（其 c 和 r 参数保持不变）。
        """
        agent = self.agents[agent_name]
        old_location, _, _ = agent.current_casino
        row, col = old_location
        move_delta = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1),   # 右
            4: (0, 0)    # 不动
        }
        d_row, d_col = move_delta[move_action]
        new_row = (row + d_row) % self.num_cells
        new_col = (col + d_col) % self.num_cells
        new_location = (new_row, new_col)
        # 在赌场列表中查找对应位置的赌场
        new_casino = None
        for casino in self.casinos:
            if casino[0] == new_location:
                new_casino = casino
                break
        if new_casino is None:
            new_casino = agent.current_casino
        agent.set_current_casino(new_casino)

    def coopration_rate(self):
        """计算全局合作率"""
        return sum(1 for agent in self.agents.values() if agent.is_cooperator) / len(self.agents)

    def each_coopration_rate(self, casino):
        """计算单个赌场的合作率"""
        n = self.get_agent_count(casino)
        n_C = self.get_cooperator_count(casino)
        return n_C / n if n != 0 else 0.0

    def render(self, t):
        """
        1) 如果 run_dir 尚未创建，则以时间戳创建文件夹；
        2) 创建或更新 visualizer，并调用其 render 方法。
        """
        if self.run_dir is None:
            root_folder = "pics"
            os.makedirs(root_folder, exist_ok=True)
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(root_folder, timestamp_folder)
            os.makedirs(self.run_dir, exist_ok=True)
        self.visualizer = PygameVisualizer(self, self.run_dir)
        self.visualizer.render(t)
