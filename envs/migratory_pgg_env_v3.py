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

    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}

    def __init__(self, network_size: int = 80, grid_division: int = 8, 
                 custom_grid_params: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None,
                 move_steps: int = 1,
                 seed: Optional[int] = None):
        super().__init__()

        self.network_size = network_size  # 晶格网络边长
        self.grid_division = grid_division  # 区域划分数
        self.grid_size = network_size // grid_division  # 每个区域的边长
        self.N = 150  # 智能体数量
        self.timestep = 0  # 时间步计数
        self.move_steps = move_steps  # 新增移动步长属性
        self.seed = seed  # 全局随机种子
        self.run_dir = None


        # 设置全局随机种子
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # 区域参数字典：允许手动传入参数；若无，则使用默认值
        self.grid_params = custom_grid_params if custom_grid_params else {
            (i, j): (1.0, 4.0) for i in range(grid_division) for j in range(grid_division)
        }

        # 生成智能体名称列表
        self.agent_names = [f"agent_{i}" for i in range(self.N)]

        # 定义动作空间和观测空间
        self.action_spaces = {agent: spaces.Tuple((spaces.Discrete(2), spaces.Discrete(5))) 
                              for agent in self.agent_names}

        self.observation_spaces = {agent: spaces.Tuple((spaces.Discrete(self.network_size),
                                                        spaces.Discrete(self.network_size)))
                                   for agent in self.agent_names}

        self.nodes = {}  # 网格节点字典
        self.agents = {}  # 智能体字典

        self.make_world()

    def make_world(self):
        self.create_network()
        self.assign_nodes_to_grids()
        self.instantiate_agents()
        self.assign_agents_to_nodes()

    def create_network(self):
        """创建周期性晶格网络，构造邻接关系"""
        for row in range(self.network_size):
            for col in range(self.network_size):
                neighbors = [
                    ((row + 1) % self.network_size, col),   # 下一个行（周期性）
                    ((row - 1) % self.network_size, col),   # 上一个行（周期性）
                    (row, (col - 1) % self.network_size),     # 左侧
                    (row, (col + 1) % self.network_size)      # 右侧
                ]
                self.nodes[(row, col)] = {"neighbors": neighbors}  # 邻接关系

    def assign_nodes_to_grids(self):
        """根据网格划分，为每个节点分配其所在网格参数"""
        for (row, col) in self.nodes:
            grid_row = row // self.grid_size
            grid_col = col // self.grid_size
            self.nodes[(row, col)]["grid_params"] = self.grid_params[(grid_row, grid_col)]

    def instantiate_agents(self):
        self.agents = {}
        for agent_id in self.agent_names:
            self.agents[agent_id] = Agent(agent_id)

    def assign_agents_to_nodes(self):
        # 1. 将所有节点按照所属网格分组
        # 注意：网格索引计算：grid_row = row // self.grid_size, grid_col = col // self.grid_size
        grid_nodes = {}  # 键为 (grid_row, grid_col)，值为该网格中所有节点（row, col）的列表
        for (row, col) in self.nodes.keys():
            grid = (row // self.grid_size, col // self.grid_size)
            grid_nodes.setdefault(grid, []).append((row, col))
        
        # 随机打乱每个网格内的节点顺序
        for grid in grid_nodes:
            random.shuffle(grid_nodes[grid])
        
        # 2. 对智能体进行分组分配
        agents_list = list(self.agents.keys())
        random.shuffle(agents_list)  # 随机化智能体顺序
        
        num_grids = len(self.grid_params)  # 通常 grid_division*grid_division，例如 64
        min_agents_per_grid = 2
        total_agents = len(agents_list)     # 例如 150
        remaining_agents = total_agents - (min_agents_per_grid * num_grids)  # 150 - 128 = 22
        
        # 先为每个网格分配最少两个智能体
        grid_assignments = {grid: [] for grid in self.grid_params.keys()}
        for grid in grid_assignments:
            # 如果网格内节点足够（每个网格一般会有 self.grid_size*self.grid_size 个节点）
            for _ in range(min_agents_per_grid):
                if grid_nodes[grid]:
                    node = grid_nodes[grid].pop()  # 从该网格随机取一个节点
                    agent = agents_list.pop(0)
                    grid_assignments[grid].append((agent, node))
        
        # 3. 将剩余的智能体随机分布到各个网格中
        grids = list(grid_assignments.keys())
        while agents_list:
            grid = random.choice(grids)
            # 从该网格中获取一个节点（如果该网格节点用完了，再随机选择其他网格）
            if not grid_nodes[grid]:
                continue
            node = grid_nodes[grid].pop()
            agent = agents_list.pop(0)
            grid_assignments[grid].append((agent, node))
        
        # 4. 更新每个智能体的当前位置
        for assignments in grid_assignments.values():
            for agent, node in assignments:
                self.agents[agent].set_current_node(node)


    def get_observation(self, agent_id):
        """返回智能体的观测信息：当前节点位置 (row, col)"""
        return self.agents[agent_id].current_node

    def global_observations(self):
        return {agent_id: self.get_observation(agent_id) for agent_id in self.agents}

    def get_reward(self, agent, game_action):
        """传入智能体和游戏动作，返回对应的奖励"""
        c, r = self.nodes[agent.current_node]["grid_params"]  # 当前节点所在网格的 (c, r) 参数
        # 根据当前节点计算所属网格的索引
        grid_row = agent.current_node[0] // self.grid_size
        grid_col = agent.current_node[1] // self.grid_size
        n = self.get_grid_agent_count(grid_row, grid_col)     # 当前网格中的总智能体数量，至少有传入智能体本身
        n_C = self.get_grid_cooperator_count(grid_row, grid_col)  # 当前网格中的合作智能体数量

        return (r * n_C) / n - c if game_action == 1 else (r * n_C) / n

    def get_grid_agent_count(self, grid_row, grid_col):
        return sum(1 for agent in self.agents.values()
                   if agent.current_node[0] // self.grid_size == grid_row
                   and agent.current_node[1] // self.grid_size == grid_col)

    def get_grid_cooperator_count(self, grid_row, grid_col):
        return sum(1 for agent in self.agents.values() if agent.is_cooperator and
                   agent.current_node[0] // self.grid_size == grid_row and
                   agent.current_node[1] // self.grid_size == grid_col)

    def reset(self, seed: Optional[int] = None):
        # 如果 reset 调用时传入新的种子，则更新全局随机种子
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.timestep = 0
        self.assign_agents_to_nodes()  # 重置智能体位置
        observations = self.global_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, Tuple[int, int]]):
        rewards = {agent: 0 for agent in self.agents.keys()}
        # 先更新合作状态及计算奖励
        for agent_id, (game_action, move_action) in actions.items():
            agent = self.agents[agent_id]
            agent.is_cooperator = (game_action == 1)
            rewards[agent_id] = self.get_reward(agent, game_action)
        # 再执行移动动作
        for agent_id, (game_action, move_action) in actions.items():
            self.move_agent(agent_id, move_action)
        observations = self.global_observations()
        self.timestep += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def move_agent(self, agent_id, move_action):
        current_node = self.agents[agent_id].current_node
        # 动作 0 表示不移动
        if move_action == 0:
            new_node = current_node
        else:
            neighbor_index = move_action - 1
            new_node = current_node
            # 根据步长重复移动
            for _ in range(self.move_steps):
                new_node = self.nodes[new_node]["neighbors"][neighbor_index]
        self.agents[agent_id].set_current_node(new_node)

    def render(self, t):
        """
        如果 run_dir 尚未创建，则以时间戳创建文件夹；
        创建或更新 visualizer，并调用其 render 方法。
        """
        if not hasattr(self, "run_dir") or self.run_dir is None:
            root_folder = "pics"
            os.makedirs(root_folder, exist_ok=True)
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(root_folder, timestamp_folder)
            os.makedirs(self.run_dir, exist_ok=True)
        self.visualizer = PygameVisualizer(self, self.run_dir)
        self.visualizer.render(t)

    def get_global_cooperation_rate(self):
        """
        计算全局合作率：所有智能体中合作智能体所占的比例。
        返回:
            float: 合作率（0-1之间），若环境中没有智能体，则返回 None。
        """
        total_agents = self.N
        if total_agents == 0:
            return None
        cooperator_count = sum(1 for agent in self.agents.values() if agent.is_cooperator)
        return cooperator_count / total_agents


    def get_grid_cooperation_rates(self):
        """
        计算每个网格区域内的合作率：
        - 遍历每个网格区域（以 self.grid_params 的键，即 (grid_row, grid_col) 为标识）
        - 对落在该区域内的智能体进行计数，若该网格中没有智能体，则对应值为 None，
            否则返回合作智能体占比。
        返回:
            dict: 键为网格区域 (grid_row, grid_col)，值为该区域内合作率或 None。
        """
        grid_coop_rates = {}
        # 遍历所有网格区域（使用 grid_params 的键）
        for grid_key in self.grid_params:
            total = 0
            coop_count = 0
            # 遍历所有智能体，判断其当前节点所属的网格区域
            for agent in self.agents.values():
                grid_row = agent.current_node[0] // self.grid_size
                grid_col = agent.current_node[1] // self.grid_size
                if (grid_row, grid_col) == grid_key:
                    total += 1
                    if agent.is_cooperator:
                        coop_count += 1
            grid_coop_rates[grid_key] = None if total == 0 else coop_count / total
        return grid_coop_rates
