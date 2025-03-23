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
    """多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game)
    
    修改要点：
      1. 观测空间仅为网格编号 (row, col)，后续将水平方向绑定增益倍率，竖直方向绑定固定成本。
      2. 每个网格预先手动绑定成本和收益参数，通过传入 custom_grids 字典实现；
      3. 每步智能体执行一个联合动作：先做博弈决策（合作或背叛），再做移动决策（上、下、左、右、不动），动作执行后直接移动到新网格，因此只维护一个 Q 表；
      4. 使用 grid 和 grids 替代 casino/casinos 的表达。
    """

    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}

    def __init__(self, 
                 L: int = 10, 
                 l: float = 1.25, 
                 N: int = 150,
                 preset_c: float = 1.0,   # 默认固定成本
                 preset_r: float = 4.0,   # 默认增益倍率
                 custom_grids: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None):
        super().__init__()
        
        # 基本参数
        self.grid_size = L                     # 总区域边长
        self.cell_size = l                     # 每个网格的边长
        self.N = N                             # 智能体数量
        self.timestep = 0                      # 时间步计数
        self.run_dir = None
        self.preset_c = preset_c
        self.preset_r = preset_r

        
        # 计算网格数量（假设正方形网格）
        self.num_cells = int(self.grid_size // self.cell_size)
        
        # 手动绑定的网格参数字典：键为 (row, col)，值为 (固定成本, 增益倍率)
        # 如果没有提供，则所有网格均采用默认值
        self.custom_grids = custom_grids if custom_grids is not None else {}
        
        # 定义联合动作空间：每个智能体的动作为 Tuple( game_action, move_action )
        # game_action: 0 (不贡献) 或 1 (贡献)
        # move_action: 0: 上, 1: 下, 2: 左, 3: 右, 4: 不动
        self.joint_action_spaces = {agent: spaces.Tuple((spaces.Discrete(2), spaces.Discrete(5))) 
                                    for agent in ["agent_" + str(i) for i in range(N)]}
        
        # 定义观测空间：观测为网格编号 (row, col)，取值分别为 Discrete(num_cells)
        self.observation_spaces = {agent: spaces.Tuple((spaces.Discrete(self.num_cells),
                                                          spaces.Discrete(self.num_cells)))
                                   for agent in self.joint_action_spaces}
        
        # 使用统一的动作空间
        self.action_spaces = self.joint_action_spaces
        
        # 初始化网格字典和智能体字典
        self.grids = {}      # 键为 (row, col)，值为 (固定成本, 增益倍率)
        self.agents = {}     # 智能体实例字典
        self.agent_names = list(self.joint_action_spaces.keys())
        
        self.make_world()

    def make_world(self):
        """创建网格和智能体"""
        self.create_grids() # 创建网格
        self.instantiate_agents() # 实例化智能体
        self.assign_agents_to_grids() # 均匀分配智能体到网格


    def create_grids(self):
        """
        创建网格，每个网格对应一个二元组 (c, r)
        - 如果 custom_grids 中有定义，则使用自定义参数；
        - 否则，所有网格均采用默认参数 preset_c 和 preset_r。
        网格的编号为 (row, col)，其中 row, col 均取值 0 ~ num_cells-1，
        其中竖直方向(row)将绑定固定成本，水平方向(col)绑定增益倍率。
        """
        for row in range(self.num_cells):
            for col in range(self.num_cells):
                pos = (row, col)
                if pos in self.custom_grids:
                    c, r = self.custom_grids[pos]
                else:
                    c, r = self.preset_c, self.preset_r
                self.grids[pos] = (c, r)

    def instantiate_agents(self):
        """实例化所有智能体，将它们存储在 self.agents 中，不进行网格分配"""
        self.agents = {}
        for agent_id in range(self.N):
            agent_name = f"agent_{agent_id}"
            self.agents[agent_name] = Agent(agent_name)

    def assign_agents_to_grids(self):
        """将已创建的智能体均匀分配到各个网格中
        分配方式：每个网格先分配 floor(N/total_grids) 个智能体，余数随机分配"""
        grid_positions = list(self.grids.keys())
        total_grids = len(grid_positions)
        base_num = self.N // total_grids  # 每个网格至少分配的数量

        # 按固定顺序分配：遍历每个网格，将 base_num 个智能体依次分配
        agent_id = 0
        for pos in grid_positions:
            for _ in range(base_num):
                agent_name = f"agent_{agent_id}"
                self.agents[agent_name].set_current_grid(pos)
                agent_id += 1

        # 处理余数：将剩余的智能体随机分配到部分网格
        remaining = self.N - agent_id
        if remaining > 0:
            random_positions = random.sample(grid_positions, remaining)
            for pos in random_positions:
                agent_name = f"agent_{agent_id}"
                self.agents[agent_name].set_current_grid(pos)
                agent_id += 1


    def get_observation(self, agent_name):
        """
        返回单个智能体的观测，观测仅为网格编号 (row, col)
        """
        agent = self.agents[agent_name]
        return agent.current_grid  # 假定 current_grid 为 (row, col)

    def get_observations(self):
        """返回所有智能体的观测字典"""
        return {agent_name: self.get_observation(agent_name) for agent_name in self.agents}

    def get_agent_count(self, grid):
        """获取指定网格中智能体数量"""
        return sum(1 for agent in self.agents.values() if agent.current_grid == grid)

    def get_cooperator_count(self, grid):
        """获取指定网格中合作者的数量"""
        return sum(1 for agent in self.agents.values() if agent.current_grid == grid and agent.is_cooperator)

    def reset(self, seed=None):
        """
        重置整个环境：
          - 重置时间步计数，
          - 重新分配智能体到网格（可选：也可保持当前分布），
          - 返回所有智能体的初始观测 (grid 编号) 和 infos。
        """
        self.timestep = 0
        self.assign_agents_to_grids()
        observations = self.get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def reset_obs(self, seed=None):
        """重置环境观测，返回所有智能体观测 (grid 编号)"""
        self.timestep = 0
        observations = self.get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, Tuple[int, int]]):
        """
        执行联合动作：
          - 每个智能体的动作为 (game_action, move_action)；
          - game_action: 0 表示不贡献，1 表示贡献；
          - move_action: 0: 上, 1: 下, 2: 左, 3: 右, 4: 不动；
        先根据 game_action 进行博弈，计算奖励；再根据 move_action 更新智能体所在网格。
        返回新的观测、奖励、终止和截断标志，以及 infos。
        """
        rewards = {agent: 0 for agent in self.agents.keys()}
        # 先处理博弈决策（合作/不合作）
        for agent_name, (game_action, move_action) in actions.items():
            agent = self.agents[agent_name]
            agent.is_cooperator = (game_action == 1)
            rewards[agent_name] = self.get_reward(agent, game_action)
        # 再处理移动决策
        for agent_name, (game_action, move_action) in actions.items():
            self.move_agent(agent_name, move_action)
        observations = self.get_observations()
        self.timestep += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def get_reward(self, agent, game_action):
        """
        奖励计算：
          - 取智能体所在网格的参数 (c, r)；
          - n 为该网格中智能体数量，n_C 为合作智能体数量；
          - 如果选择合作，奖励 = (r * n_C)/n - c；否则奖励 = (r * n_C)/n。
        """
        grid = agent.current_grid  # (row, col)
        c, r = self.grids[grid]
        n = self.get_agent_count(grid)
        n_C = self.get_cooperator_count(grid)
        if n == 0:
            return 0
        if game_action == 1:
            reward = (r * n_C) / n - c
        else:
            reward = (r * n_C) / n
        return reward

    def move_agent(self, agent_name, move_action: int):
        """
        离散环状移动：
          - 当前网格为 (row, col)；
          - 移动动作定义：0: 上 (row+1), 1: 下 (row-1), 2: 左 (col-1), 3: 右 (col+1), 4: 不动；
          - 使用模运算实现边界环绕。
        """
        agent = self.agents[agent_name]
        row, col = agent.current_grid
        # 定义移动的变化（注意：行数增加表示向上）
        move_delta = {
            0: (1, 0),    # 上
            1: (-1, 0),   # 下
            2: (0, -1),   # 左
            3: (0, 1),    # 右
            4: (0, 0)     # 不动
        }
        d_row, d_col = move_delta[move_action]
        new_row = (row + d_row) % self.num_cells
        new_col = (col + d_col) % self.num_cells
        agent.set_current_grid((new_row, new_col))

    def coopration_rate(self):
        """计算整体合作率"""
        return sum(1 for agent in self.agents.values() if agent.is_cooperator) / len(self.agents)
    
    def each_coopration_rate(self, grid):
        """计算指定网格的合作率"""
        n = self.get_agent_count(grid)
        n_C = self.get_cooperator_count(grid)
        return n_C / n if n != 0 else 0.0

    def render(self, t):
        """
        如果 run_dir 尚未创建，则以时间戳创建文件夹；
        创建或更新 visualizer，并调用其 render 方法。
        """
        if self.run_dir is None:
            root_folder = "pics"
            os.makedirs(root_folder, exist_ok=True)
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(root_folder, timestamp_folder)
            os.makedirs(self.run_dir, exist_ok=True)
        self.visualizer = PygameVisualizer(self, self.run_dir)
        self.visualizer.render(t)
