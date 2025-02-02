from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

class MigratoryPGGEnv(ParallelEnv):
    """ 多智能体迁徙公共物品博弈环境 (Migratory Public Goods Game) """
    
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_env"}
    
    def __init__(self, L=5, l=1, r_min=1.5, r_max=4.0, N=150):
        super().__init__()
        
        # 关键参数定义
        self.grid_size = (L, L)  # 总区域大小
        self.casino_size = l     # 每个赌场边长
        self.r_params = (r_min, r_max) # 增益因子范围
        self.N = N # 智能体数量
        
        # 定义观测/动作空间（PettingZoo核心要求）
        self.agents = ["agent_" + str(i) for i in range(N)]
        self.action_spaces = {agent: spaces.Discrete(2) for agent in self.agents}  # 0: 不贡献, 1: 贡献
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32) for agent in self.agents}

    def reset(self):
        """重置环境到初始状态。"""
        self.agents = self.initialize_agents()
        self.state = self.get_pre_game_observation()  # 初始状态为博弈前的观测
        return self.state

    def step(self, actions):
        """接收智能体的动作，更新环境状态。"""
        rewards = []
        for agent, action in zip(self.agents, actions):
            reward = self.get_reward(agent, action)
            rewards.append(reward)
            self.move_agents(agent, action)
        
        # 更新为博弈后的观测状态
        self.state = self.get_post_game_observation()
        done = self.check_done()  # 检查是否结束
        return self.state, rewards, done

    def get_pre_game_observation(self):
        """获取博弈前的观测状态。"""
        observations = []
        for agent in self.agents:
            m = agent.current_casino  # 当前赌场
            c, alpha = m  # 成本参数和增益因子
            n = self.get_agent_count(m)  # 当前赌场的智能体数量
            observations.append((c, alpha, n))
        return observations

    def get_post_game_observation(self):
        """获取博弈后的观测状态。"""
        observations = []
        for agent in self.agents:
            m = agent.current_casino  # 当前赌场
            c, alpha = m  # 成本参数和增益因子
            n = self.get_agent_count(m)  # 当前赌场的智能体数量
            n_C = self.get_cooperator_count(m)  # 当前赌场的合作者数量
            observations.append((c, alpha, n, n_C))
        return observations

    def get_reward(self, agent, action):
        """计算并返回智能体在当前状态下的奖励。"""
        m = agent.current_casino
        c, alpha = m  # 从赌场索引中获取成本和增益因子
        n = self.get_agent_count(m)
        n_C = self.get_cooperator_count(m)
        
        # 计算增益因子 r
        r = self.r_params[0] + (self.r_params[1] - self.r_params[0]) * alpha
        
        # 计算奖励
        if action == 1:  # 如果智能体选择合作
            reward = (r * n_C) / n - c
        else:  # 如果智能体选择背叛
            reward = (r * n_C) / n
        
        return reward

    def check_done(self):
        """检查是否结束。"""
        pass  # 当前不执行任何操作

    def initialize_agents(self):
        """初始化智能体并均匀分布到赌场。"""
        agents = []
        for i in range(self.N):
            agent = self.create_agent(i)  # 创建智能体
            agents.append(agent)
        return agents

    def create_agent(self, agent_id):
        """创建一个智能体。"""
        pass  # 当前不执行任何操作

    def get_agent_count(self, casino):
        """获取指定赌场的智能体数量。"""
        return sum(1 for agent in self.agents if agent.current_casino == casino)

    def get_cooperator_count(self, casino):
        """获取指定赌场的合作者数量。"""
        return sum(1 for agent in self.agents if agent.current_casino == casino and agent.is_cooperator)

    def move_agents(self, agent, action):
        """根据智能体的迁移策略更新其在赌场中的位置。"""
        pass  # 具体实现根据迁移策略

    def render(self):
        """可视化环境状态（可选）。"""
        pass


