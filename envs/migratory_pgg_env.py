from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from agents.agent import Agent
import random
import numpy as np
from typing import List, Type, Union
from envs.visualizer import PygameVisualizer

class MigratoryPGGEnv(ParallelEnv):
    """ 多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game) """
    
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}
    
    def __init__(self, L=5, l=1, r_min=1.5, r_max=4.0, N=75):
        super().__init__()
        
        # 关键参数定义
        self.grid_size = L   # 总区域边长大小
        self.casino_size = l     # 每个赌场边长
        self.r_params = (r_min, r_max) # 增益因子范围
        self.r_params = (float(r_min), float(r_max))
        self.casinos = []  # 赌场列表
        self.N = N # 智能体数量  
        self.phase: str = "pre_game"  # 预设为博弈前阶段
        self.step_size = 1.0 / (self.grid_size // self.casino_size)  # 赌场之间的步长

        self.casinos = []  # 赌场列表
        self.agent_names = ["agent_" + str(i) for i in range(N)]  # 仅存储智能体编号
        self.agents = {}  # 存储智能体实例的字典

        self.timestep = 0 # 时间步计数


        # 定义动作空间
        self.game_action_spaces = {agent: spaces.Discrete(2) for agent in self.agent_names}  #字典 0: 不贡献, 1: 贡献 
        self.move_action_spaces = {agent: spaces.Discrete(5) for agent in self.agent_names}  #字典 0: 上, 1: 下, 2: 左, 3: 右, 4: 不动

        # 博弈后的观测空间
        self.post_game_observation_spaces = {
            agent: spaces.Tuple((
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 成本参数 c
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 增益参数 alpha
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32),  # 赌场的智能体数目，最大可能性为 N
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32)   # 赌场的合作智能体数目，最大可能性为 N
            )) for agent in self.agent_names
        }
        # 博弈前的观测空间
        self.pre_game_observation_spaces = {
            agent: spaces.Tuple((
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 成本参数 c
                spaces.Box(low=0.01, high=1.0, shape=(), dtype=np.float32),  # 增益参数 alpha
                spaces.Box(low=0, high=self.N, shape=(), dtype=np.int32)  # 赌场的智能体数目，最大可能性为 N
            )) for agent in self.agent_names
        }

        # 预设观测空间和动作空间
        self.phase = "pre_game"
        self.observation_spaces = self.pre_game_observation_spaces
        self.action_spaces = self.game_action_spaces

        # 调用 make_world() 来创建赌场和智能体
        self.make_world()

    def make_world(self):
        """ 创建赌场 & 智能体 """
        self.create_casinos()
        self.create_agents()

    def create_casinos(self):
        """ 创建赌场，每个赌场是一个二元数组 (c, alpha) """
        num_casinos_per_row = self.grid_size // self.casino_size

        for row in range(num_casinos_per_row):
            for col in range(num_casinos_per_row):
                c = (row + 1) * self.step_size
                alpha = (col + 1) * self.step_size
                self.casinos.append((c, alpha))

    def create_agents(self):
        """实例化智能体，每个智能体是一个 Agent 类，并均匀分配到赌场"""
        self.agents = {}  # 以字典存储智能体
        total_casinos = len(self.casinos)

        agent_id = 0
        # 均匀分配智能体到赌场
        for casino in self.casinos:
            for _ in range(self.N // total_casinos):
                agent_name = f"agent_{agent_id}"
                self.agents[agent_name] = Agent(agent_name)  # 先创建智能体
                self.agents[agent_name].set_current_casino(casino)  # 绑定赌场
                agent_id += 1

        # 处理余数，随机分配到赌场
        random_casinos = random.sample(self.casinos, self.N % total_casinos)
        for casino in random_casinos:
            agent_name = f"agent_{agent_id}"
            self.agents[agent_name] = Agent(agent_name)
            self.agents[agent_name].set_current_casino(casino)
            agent_id += 1

            
    def get_observation_space(self, agent_name):
        """返回单个智能体的观测空间"""
        
        agent = self.agents[agent_name]  # 直接获取该智能体对象
        
        if self.phase == "pre_game":
            m = agent.current_casino
            c, alpha = m
            n = self.get_agent_count(m)
            return (c, alpha, n)  # 返回元组
        else:
            m = agent.current_casino
            c, alpha = m
            n = self.get_agent_count(m)
            n_C = self.get_cooperator_count(m)
            return (c, alpha, n, n_C)  # 返回元组
        
    def get_observation_spaces(self, phase):
        """返回所有智能体的观测空间, 统一管理"""
        if phase == "pre_game":
            return self.get_pre_game_observation()
        else:
            return self.get_post_game_observation()
    
    def get_pre_game_observation(self):
        """获取博弈前的全局观测状态, 返回字典 {agent_name: obs}"""
        return {agent_name: self.get_observation_space(agent_name) for agent_name in self.agents}


    def get_post_game_observation(self):
        """获取博弈后的全局观测状态, 返回字典 {agent_name: obs}"""
        return {agent_name: self.get_observation_space(agent_name) for agent_name in self.agents}
        

    def get_agent_count(self, casino):
        """获取指定赌场的智能体数量。"""
        return sum(1 for agent in self.agents.values() if agent.current_casino == casino)

    def get_cooperator_count(self, casino):
        """获取指定赌场的合作者数量。"""
        return sum(1 for agent in self.agents.values() if agent.current_casino == casino and agent.is_cooperator)

    
    # action_space, valid_actions = env.action_space(agent_name) 用两个值接收返回值
    # action_idx = np.random.randint(0, action_space.n)   0, 1, or 2 随机选择可选动作
    # action = possible_moves[action_idx] 根据随机选择动作索引，获取实际动作

    def get_action_spaces(self):
        """返回对应阶段的离散动作空间"""
        return spaces.Discrete(2) if self.phase == "pre_game" else spaces.Discrete(5) # 后续需要过滤非法动作
    

    def valid_actions(self, agent_name):
        """代理调用 Agent 实例的合法动作"""
        return self.agents[agent_name].valid_moves(self.step_size)

    # action_space = env.action_space(agent)  # 一直是 Discrete(n)
    # valid_actions = env.get_valid_actions(agent)  # 取合法动作

    # # 选择动作索引
    # action_idx = np.random.randint(0, len(valid_actions))
    # actual_action = valid_actions[action_idx]  # 映射到真实动作

    def reset(self, seed=None):
        """重置环境到初始状态"""
        self.phase = "pre_game"

        # 重新初始化时间步计数
        self.timestep = 0

        # 生成符合 PettingZoo 规范的 observations
        observations = self.get_pre_game_observation()

        self.observation_spaces = self.pre_game_observation_spaces
        self.action_spaces = self.game_action_spaces

        infos = {agent: {} for agent in self.agents.keys()}  # PettingZoo 规范
        return observations, infos # 返回观测空间的字典



    def step(self, actions):
        """根据当前阶段处理不同类型的决策"""
        rewards = {agent: 0 for agent in self.agents.keys()}  # 遍历智能体 ID
        if self.phase == "pre_game":
            # 处理贡献决策
            for agent_name, game_action in actions.items():
                agent = self.agents[agent_name]  # 直接通过字典获取智能体对象
                agent.is_cooperator = (game_action == 1)
                rewards[agent_name] = self.get_reward(agent, game_action) # 计算奖励

            # 进入博弈后阶段
            self.phase = "post_game"
            observations = self.get_post_game_observation()  # 直接返回字典 {agent: obs}
            self.observation_spaces = self.post_game_observation_spaces
            self.action_spaces = self.move_action_spaces

        else:
            for agent_name, move_action in actions.items():
                agent = self.agents[agent_name]
                # old_position = agent.current_casino  # 记录移动前的位置

                self.move_agents(agent_name, move_action)  # 执行移动

                # new_position = agent.current_casino  # 移动后的新位置
                # print(f"{agent_name} 从 {old_position} 移动到 {new_position}，执行动作 {move_action}")


            # 重新进入博弈前阶段
            self.phase = "pre_game"
            observations = self.get_pre_game_observation()  # 直接返回字典 {agent: obs}
            self.observation_spaces = self.pre_game_observation_spaces
            self.action_spaces = self.game_action_spaces

        terminations = {agent: False for agent in self.agents.keys()}
        truncations = {agent: False for agent in self.agents.keys()}
        infos = {agent: {} for agent in self.agents.keys()}

        return observations, rewards, terminations, truncations, infos



    def get_reward(self, agent, game_action):
        """计算并返回智能体在当前状态下的奖励。"""

        m = agent.current_casino
        c, alpha = m  # 从赌场索引中获取成本和增益因子
        n = self.get_agent_count(m)
        n_C = self.get_cooperator_count(m)
        # 计算增益因子 r
        r = self.r_params[0] + (self.r_params[1] - self.r_params[0]) * alpha
        
        
        # 计算奖励
        if game_action == 1:  # 如果智能体选择合作
            reward = (r * n_C) / n - c
        else:  # 如果智能体选择背叛
            reward = (r * n_C) / n

        return reward

    def check_done(self):
        """检查是否结束。"""
        pass  # 当前不执行任何操作


    def move_agents(self, agent_name, move_action):
        """ 根据智能体的迁移策略更新其赌场位置 """
        
        agent = self.agents[agent_name]  # 获取智能体实例
        old_x, old_y = agent.current_casino  # 获取当前赌场位置

        # 预定义移动增量映射 {上, 下, 左, 右, 不动}
        move_delta = {
            0: (0, self.step_size),   # 上移
            1: (0, -self.step_size),  # 下移
            2: (-self.step_size, 0),  # 左移
            3: (self.step_size, 0),   # 右移
            4: (0, 0)            # 不动
        }

        # 直接获取新的位置，无需额外边界检查
        dx, dy = move_delta[move_action]  
        new_x, new_y = old_x + dx, old_y + dy

        # 更新智能体的位置
        agent.set_current_casino((new_x, new_y))

    def coopration_rate(self):
        """计算合作率"""
        return sum(1 for agent in self.agents.values() if agent.is_cooperator) / len(self.agents)

    def render(self):
        """调用外部可视化类""" # TODO
        self.visualizer.render()








