from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from agents.agent import Agent
import random
import numpy as np
from typing import List, Type, Union
from envs.visualizer import PygameVisualizer
import os
from datetime import datetime

class MigratoryPGGEnv(ParallelEnv):
    """ 多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game) """
    
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}
    
    def __init__(self, L=5, l=1, r_min=1.5, r_max=6.0, N=150):
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

        self.agent_names = ["agent_" + str(i) for i in range(N)]  # 仅存储智能体编号
        self.agents = {}  # 存储智能体实例的字典

        self.timestep = 0 # 时间步计数

        self.run_dir = None


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

    # def create_casinos(self):
    #     """ 创建赌场并随机打乱，每个赌场是一个二元数组 (c, alpha) """
    #     num_casinos_per_row = self.grid_size // self.casino_size
    #     coords = []
    #     for row in range(num_casinos_per_row):
    #         for col in range(num_casinos_per_row):
    #             c = (row + 1) * self.step_size
    #             alpha = (col + 1) * self.step_size
    #             coords.append((c, alpha))

    #     random.shuffle(coords)
    #     self.casinos = coords


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


    def reset_obs(self, seed=None):
        """重置环境观测"""
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
                self.move_agents(agent_name, move_action)  # 执行移动

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
        """
        离散环状世界:
        - 坐标只能是 step_size, 2*step_size, ..., 1.0
        - 若在 1.0 处继续向外移, 则绕回 step_size
        - 同理在 step_size 处往反方向移动则回到 1.0
        """
        agent = self.agents[agent_name]
        old_x, old_y = agent.current_casino  # 离散坐标, 比如 0.2/0.4/1.0 等

        # 如果尚未构建 coords 和 coord_to_idx，可以在 __init__ 或 create_casinos() 里一次性完成
        # 这里演示写在 move_agents() 中, 也行, 但最好只执行一次
        step_size = self.step_size
        n = int(round(1.0 / step_size))  # 如果 step_size=0.2 => n=5
        # 构建离散坐标列表
        coords = [step_size * (i + 1) for i in range(n)]  # [0.2, 0.4, 0.6, 0.8, 1.0]
        # 建立 "坐标 -> 索引" 字典, 避免用 list.index() 时出现浮点比较
        coord_to_idx = {val: i for i, val in enumerate(coords)}

        # 定义动作 -> 在 x,y 上的“索引增量”
        # move_action: 0=上,1=下,2=左,3=右,4=不动
        # x_idx+1 => 右移一格, x_idx-1 => 左移, y_idx+1 => 上移, y_idx-1 => 下移
        move_delta_idx = {
            0: (0, +1),   # 上
            1: (0, -1),   # 下
            2: (-1, 0),   # 左
            3: (+1, 0),   # 右
            4: (0,  0)    # 不动
        }

        dx_idx, dy_idx = move_delta_idx[move_action]

        # 找到 old_x, old_y 的“索引”
        old_x_idx = coord_to_idx[old_x]
        old_y_idx = coord_to_idx[old_y]

        # 计算新索引 (mod n) => 环状
        new_x_idx = (old_x_idx + dx_idx) % n
        new_y_idx = (old_y_idx + dy_idx) % n

        # 映射回浮点坐标
        new_x = coords[new_x_idx]  
        new_y = coords[new_y_idx]

        # 更新智能体的位置 (仍是离散值)
        agent.set_current_casino((new_x, new_y))




    def coopration_rate(self):
        """计算合作率"""
        return sum(1 for agent in self.agents.values() if agent.is_cooperator) / len(self.agents)
    
    def each_coopration_rate(self, casino):
        """计算每个赌场的合作率"""
        n = self.get_agent_count(casino)
        n_C = self.get_cooperator_count(casino)
        return n_C / n if n != 0 else 0.0  # 赌场为空时返回 0.0


    def render(self, t):
        """
        1) 如果本环境还没有 run_dir，第一次调用时创建带时间戳的文件夹
        2) 创建或更新 self.visualizer
        3) 调用 self.visualizer.render(t)
        """
        if self.run_dir is None:
            # 只在第一次调用时进来
            root_folder = "pics"
            os.makedirs(root_folder, exist_ok=True)

            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(root_folder, timestamp_folder)
            os.makedirs(self.run_dir, exist_ok=True)

        
        self.visualizer = PygameVisualizer(self, self.run_dir)

        # 调用可视化器的绘制方法
        self.visualizer.render(t)








