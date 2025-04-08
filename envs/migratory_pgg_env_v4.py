# 导入所需的库
from pettingzoo import ParallelEnv  # PettingZoo提供的并行环境基类
from gymnasium import spaces         # Gymnasium中的空间定义，用于定义动作空间和观测空间
import numpy as np                   # 数值计算库，主要用于数组和数学运算
import random                        # Python内置的随机数库
from typing import Dict, Tuple, Optional  # 类型提示工具，用于标注变量的类型
from envs.visualizer import PygameVisualizer  # 自定义的可视化模块，用于在pygame中绘制环境
import os
from datetime import datetime        # 获取当前时间的模块
from gymnasium.utils import seeding  # 用于环境随机种子的设置

# 定义一个带有周期边界的公共物品博弈环境类，所有智能体在二维空间中移动，并与邻居进行博弈。
class MigratoryPGGEnv(ParallelEnv):
    # 定义环境元数据，包括支持的渲染模式和环境名称
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_v4"}

    def __init__(self, N=20, max_cycles=500, size=100, speed=1.0, radius=10.0,
                 cost=1.0, r=1.5, beta=0.5, render_mode=None, visualize=False, seed=None):
        """
        初始化环境参数
        
        参数:
        - N: 智能体的数量
        - max_cycles: 环境运行的最大步数（周期）
        - size: 二维空间的边长（环境大小）
        - speed: 每个智能体的移动速度
        - radius: 邻居判定的半径（在这个范围内的智能体视为邻居）
        - cost: 合作策略的成本
        - r: 公共物品博弈中贡献的乘数因子
        - beta: 用于策略更新时的转移概率参数（影响策略改变的概率）
        - render_mode: 渲染模式（目前支持 "human"）
        - visualize: 是否启用可视化（True时使用PygameVisualizer）
        - seed: 随机种子，用于保证实验可重复性
        """
        super().__init__()
        self.N = N
        self.max_cycles = max_cycles
        self.size = size
        self.speed = speed
        self.radius = radius
        self.cost = cost
        self.r = r
        self.beta = beta
        self.render_mode = render_mode
        self.visualize = visualize

        # 为每个智能体生成唯一标识，如 "agent_0", "agent_1", ..., "agent_N-1"
        self.possible_agents = [f"agent_{i}" for i in range(self.N)]
        self.agents = list(self.possible_agents)  # 拷贝一份作为当前存活列表

        # self.num_agents = len(self.agents)
        # self.max_num_agents = self.num_agents

        # 设置随机种子，保证实验结果可重复
        self._seed(seed)
        # 初始化每个智能体的动作空间和观测空间
        self._init_spaces()
        # 如果需要可视化，则创建一个 PygameVisualizer 实例
        self.visualizer = PygameVisualizer(self.size, self.size) if visualize else None
        # 重置环境，初始化所有状态
        self.reset()

    def _seed(self, seed=None):
        """
        设置环境的随机种子
        
        参数:
        - seed: 用户指定的随机种子
        
        返回:
        - 返回实际使用的随机种子
        """
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return seed

    def _init_spaces(self):
        """
        初始化每个智能体的观测空间和动作空间
        
        观测空间定义:
        - "self": 当前智能体自身的状态，包括位置、速度、策略（合作或背叛）、以及上一次博弈的收益
        - "neighbors": 周围邻居的信息，包含与 "self" 相同的状态数据，使用 Sequence 来容纳不同数量的邻居
        
        动作空间:
        - 定义为二维向量，范围在 -1.0 到 1.0，表示智能体的移动方向和大小（会经过归一化处理）
        """
        self.observation_spaces = {
            agent: spaces.Dict({
                "self": spaces.Dict({
                    "strategy": spaces.Discrete(2),
                    "last_payoff": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                }),
                "neighbors": spaces.Sequence(spaces.Dict({
                    "relative_position": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "distance": spaces.Box(low=0.0, high=self.radius, shape=(), dtype=np.float32),
                    "strategy": spaces.Discrete(2),
                    "last_payoff": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
                }))
            }) for agent in self.agents
        }


        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }
    
    def reset(self, seed=None, options=None):
        """
        重置环境，初始化所有智能体的状态
        
        参数:
        - seed: 可选参数，若提供则重新设置随机种子
        - options: 其他选项（当前未使用）
        
        初始化内容:
        - timestep: 当前步数重置为0
        - pos: 每个智能体在二维空间中的随机位置
        - vel: 每个智能体的初始速度（随机方向，固定速率）
        - strategy: 每个智能体的初始策略（随机选取0或1）
        - last_payoff: 每个智能体上一次的收益初始化为0.0
        
        返回:
        - 环境中所有智能体的初始观测信息
        """
        if seed is not None:
            self._seed(seed)

        self.timestep = 0
        self.pos = {}
        self.vel = {}
        self.strategy = {}
        self.last_payoff = {}

        self.agents = list(self.possible_agents)
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}



        for agent in self.agents:
            self.pos[agent] = self.np_random.random(2) * self.size
            theta = self.np_random.random() * 2 * np.pi
            self.vel[agent] = self.speed * np.array([np.cos(theta), np.sin(theta)])
            self.last_payoff[agent] = 0.0

        # 严格控制一半智能体合作，一半背叛
        shuffled_agents = list(self.agents)  # 转换为 list 再 shuffle
        self.np_random.shuffle(shuffled_agents)

        
        half = (self.N + 1) // 2
        cooperators = set(shuffled_agents[:half])
        for agent in self.agents:
            self.strategy[agent] = 1 if agent in cooperators else 0

        return self._compute_observations(), self.infos


    def step(self, actions: Dict[str, np.ndarray]):
        """
        根据所有智能体的动作更新环境状态
        
        参数:
        - actions: 字典，键为智能体标识，值为对应的动作（二维向量）
        
        过程:
        1. 更新步数（timestep）
        2. 根据动作更新每个智能体的速度和位置（使用归一化后的移动方向，保证速度大小为固定值）
           注意：使用周期边界，即位置超过边界时从另一侧进入（模运算实现）
        3. 计算所有智能体在当前步的博弈收益：
           - 对于每个智能体，根据其邻居（在指定半径内）组成小组进行博弈
           - 每个小组中，所有智能体的贡献相加后乘以倍增因子 r 得到公共池
           - 公共池中的收益平分给小组中的每个智能体，每个合作者还需要扣除合作成本
        4. 策略更新：
           - 每个智能体随机选择一个邻居，与其比较收益差异，根据 sigmoid 函数计算改变策略的概率
           - 若随机数小于该概率，则智能体采用邻居的策略
        5. 更新上一次博弈收益，并计算新的观测信息
        
        返回:
        - obs: 更新后的观测信息
        - all_payoffs: 每个智能体在当前步的博弈收益
        - terminations: 表示每个智能体是否终止（基于最大步数）
        - truncated: 同 terminations，当前设计中两者一致
        - infos: 附加信息（目前为空字典）
        """
        self.timestep += 1

        # 根据智能体动作更新速度和位置
        for agent, move in actions.items():
            # 对动作向量进行归一化（防止除以0，加上极小值1e-8）
            move = move / (np.linalg.norm(move) + 1e-8)
            # 更新速度方向，幅度为 speed
            self.vel[agent] = move * self.speed
            # 更新位置，并通过取模运算实现周期性边界条件
            self.pos[agent] = (self.pos[agent] + self.vel[agent]) % self.size

        # 初始化所有智能体的收益为0
        all_payoffs = {agent: 0.0 for agent in self.agents}

        # 遍历每个智能体，计算博弈收益
        for agent in self.agents:
            # 获取当前智能体的邻居列表
            neighbors = self._get_neighbors(agent)
            # 小组包括当前智能体和所有邻居
            group = [agent] + neighbors
            # 计算小组中所有合作智能体的贡献总和（合作策略为1时贡献为 cost）
            contribs = sum(self.strategy[a] * self.cost for a in group)
            # 将总贡献乘以倍增因子 r 得到公共池
            pool = contribs * self.r
            # 平均分配公共池中的收益到小组内每个智能体
            share = pool / len(group)
            # 为小组中每个智能体计算收益：平分收益减去（若合作则扣除 cost）
            for a in group:
                payoff = share - (self.cost if self.strategy[a] == 1 else 0.0)
                all_payoffs[a] += payoff

        # 策略更新：每个智能体随机选择一个邻居比较收益，根据概率决定是否采用邻居的策略
        for agent in self.agents:
            neighbors = self._get_neighbors(agent)
            # 如果没有邻居则跳过策略更新
            if not neighbors:
                continue
            # 随机选择一个邻居
            other = self.np_random.choice(neighbors)
            # 计算收益差（邻居收益 - 自身收益）
            delta = all_payoffs[other] - all_payoffs[agent]
            # 根据sigmoid函数计算改变策略的概率（beta 控制敏感度）
            prob = 1 / (1 + np.exp(-self.beta * delta))
            # 如果随机数小于计算出的概率，则智能体采用邻居的策略
            if self.np_random.random() < prob:
                self.strategy[agent] = self.strategy[other]

        # 保存本次计算的收益作为上一次收益
        self.last_payoff = all_payoffs.copy()
        # 计算当前所有智能体的观测信息
        obs = self._compute_observations()

        # 判断是否达到最大步数，若达到则终止
        truncated = {agent: self.timestep >= self.max_cycles for agent in self.agents}
        terminations = truncated.copy()
        infos = {agent: {} for agent in self.agents}

        # 返回观测、收益、终止标志、截断标志和附加信息

        # 更新活跃的智能体列表（PettingZoo 强制要求）
        self.agents = [
            agent for agent in self.possible_agents
            if not terminations[agent] and not truncated[agent]]

        return obs, all_payoffs, terminations, truncated, infos
    
    def _get_neighbors(self, agent):
        """
        获取指定智能体的邻居
        
        参数:
        - agent: 指定的智能体标识
        
        邻居判定:
        - 计算其他每个智能体与当前智能体之间的距离（考虑周期性边界）
        - 若距离小于等于指定半径 radius，则视为邻居
        
        返回:
        - 邻居列表（包含满足条件的其他智能体标识）
        """
        agent_pos = self.pos[agent]
        neighbors = []
        for other in self.agents:
            if other == agent:
                continue
            # 正确的周期性边界下的最短距离计算
            delta = np.abs(self.pos[other] - agent_pos)
            delta = np.minimum(delta, self.size - delta)
            dist = np.linalg.norm(delta)
            if dist <= self.radius:
                neighbors.append(other)
        return neighbors


    def _compute_observations(self):
        """
        计算所有智能体的观测信息
        
        对于每个智能体:
        - "self": 自身的状态信息（位置、速度、策略、上一次收益）
        - "neighbors": 邻居的状态信息（列表形式，每个邻居包含相同的属性）
        
        返回:
        - 包含每个智能体观测信息的字典
        """
        obs = {}
        for agent in self.agents:
            agent_pos = self.pos[agent]
            neighbors = self._get_neighbors(agent)
            
            neighbor_data = []
            for n in neighbors:
                delta = self.pos[n] - agent_pos
                delta = (delta + self.size / 2) % self.size - self.size / 2  # 周期性处理
                dist = np.linalg.norm(delta)
                relative_position = delta / (dist + 1e-8)  # 单位方向向量

                neighbor_data.append({
                    "relative_position": relative_position.astype(np.float32),
                    "distance": float(dist),
                    "strategy": int(self.strategy[n]),
                    "last_payoff": float(self.last_payoff[n]),
                })

            obs[agent] = {
                "self": {
                    "strategy": int(self.strategy[agent]),
                    "last_payoff": float(self.last_payoff[agent])
                },
                "neighbors": neighbor_data
            }

        return obs
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def state(self):
        """
        返回全局状态，格式为 ndarray，用于 centralized training。
        拼接所有智能体的策略、收益、位置和速度信息。
        每个智能体 6 个特征：strategy, payoff, pos(x,y), vel(x,y)
        """
        state_list = []
        for agent in self.agents:
            s = float(self.strategy[agent])
            p = float(self.last_payoff[agent])
            pos = self.pos[agent].astype(np.float32)  # (2,)
            vel = self.vel[agent].astype(np.float32)  # (2,)
            state_list.append(np.concatenate([[s, p], pos, vel]))  # shape: (6,)
        
        return np.concatenate(state_list, axis=0)  # shape: (N * 6,)


    def render(self):
        """
        渲染当前环境状态
        
        如果启用了可视化，则调用 visualizer 绘制所有智能体的状态（位置、速度和策略）
        """
        if self.visualizer:
            self.visualizer.draw(self.pos, self.vel, self.strategy)

    def close(self):
        """
        关闭渲染窗口
        
        如果启用了可视化，则调用 visualizer 的 close 方法关闭窗口
        """
        if self.visualizer:
            self.visualizer.close()


########### 工具函数 ###########

    def cooperation_rate(self):
        """
        计算合作率，即所有智能体中采用合作策略（值为1）的比例
        
        返回:
        - 合作率（0~1之间的浮点数）
        """
        return sum(self.strategy[a] for a in self.agents) / self.N

    def average_payoff(self):
        """
        计算所有智能体的平均收益
        
        返回:
        - 平均收益（浮点数）
        """
        return np.mean([self.last_payoff[a] for a in self.agents])

    def total_payoff(self):
        """
        计算所有智能体的总收益
        
        返回:
        - 总收益（浮点数）
        """
        return np.sum([self.last_payoff[a] for a in self.agents])

    def average_cooperator_payoff(self):
        """
        计算合作智能体的平均收益
        
        过程:
        - 筛选出所有采用合作策略（值为1）的智能体
        - 如果没有合作智能体，则返回0.0
        - 否则计算这些智能体的平均收益
        
        返回:
        - 合作智能体的平均收益（浮点数）
        """
        cooperators = [a for a in self.agents if self.strategy[a] == 1]
        if not cooperators:
            return 0.0
        return np.mean([self.last_payoff[a] for a in cooperators])

    def average_defector_payoff(self):
        """
        计算背叛智能体的平均收益
        
        过程:
        - 筛选出所有采用背叛策略（值为0）的智能体
        - 如果没有背叛智能体，则返回0.0
        - 否则计算这些智能体的平均收益
        
        返回:
        - 背叛智能体的平均收益（浮点数）
        """
        defectors = [a for a in self.agents if self.strategy[a] == 0]
        if not defectors:
            return 0.0
        return np.mean([self.last_payoff[a] for a in defectors])

    def cooperation_clustering(self):
        """
        计算合作智能体之间的聚集程度
        
        过程:
        - 收集所有合作智能体的位置信息
        - 若合作智能体数量小于等于1，则聚集程度返回0.0
        - 否则计算所有合作智能体之间两两的欧式距离均值
        - 以均值的倒数（加上一个小常数1e-6以防除0）作为聚集程度指标
        
        返回:
        - 聚集程度（值越大表示合作智能体聚集越紧密）
        """
        coop_positions = [self.pos[a] for a in self.agents if self.strategy[a] == 1]
        if len(coop_positions) <= 1:
            return 0.0
        dists = [np.linalg.norm(p1 - p2) for i, p1 in enumerate(coop_positions) for j, p2 in enumerate(coop_positions) if i < j]
        return 1 / (np.mean(dists) + 1e-6)
