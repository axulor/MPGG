import gymnasium as gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
import argparse 
from typing import List, Tuple, Dict
import random
from gymnasium.utils import seeding  

# ==============================================================================
# == Agent Class Definition  ==
# ==============================================================================
class Agent:
    def __init__(self):
        self.id = None      # int
        self.name = None    # 字符串
        self.position = np.zeros(2, dtype=np.float32) # 位置
        self.direction_vector = np.zeros(2, dtype=np.float32) # 方向向量
        self.strategy = np.array([0], dtype=np.int32) # 0 背叛者, 1 合作者
        self.current_payoff = np.array([0.0], dtype=np.float32) # 收益

# ==============================================================================
# == MultiAgentGraphEnv Class Definition  ==
# ==============================================================================
class MultiAgentGraphEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, args: argparse.Namespace): # 类型为 argparse.Namespace
        super().__init__()
        # 参数初始化
        self.num_agents = args.num_agents
        self.world_size = args.world_size
        self.speed = args.speed
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r
        self.beta = args.beta
        self.env_max_steps = args.env_max_steps           # 环境最大步数限制

        self.seed_val = args.seed 
        self.seed(args.seed)


        # 初始化智能体列表
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # 环境自身属性
        self.current_steps = 0 # 标记执行的步数计数
        self.dist_adj = None # 距离邻接矩阵
        

        # 配置 Gym 空间
        self.agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = [self.agent_action_space] * self.num_agents

        temp_agent = Agent() 
        temp_agent.position = np.zeros(2); temp_agent.direction_vector = np.zeros(2)
        temp_agent.strategy = np.array([0]); temp_agent.current_payoff = np.array([0.0])
        obs_sample = self._get_agent_obs(temp_agent) 
        obs_dim = obs_sample.shape[0]

        self.agent_obs_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = [self.agent_obs_space] * self.num_agents


        node_obs_dim_tuple = (self.num_agents, obs_dim) 
        adj_dim_tuple = (self.num_agents, self.num_agents)
        agent_id_dim_tuple = (1,)
        edge_dim_tuple = (1,) 

        self.node_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=node_obs_dim_tuple, dtype=np.float32)] * self.num_agents
        self.adj_observation_space = [spaces.Box(low=0, high=+np.inf, shape=adj_dim_tuple, dtype=np.float32)] * self.num_agents
        self.agent_id_observation_space = [spaces.Box(low=0, high=self.num_agents - 1, shape=agent_id_dim_tuple, dtype=np.int32)] * self.num_agents
        self.share_agent_id_observation_space = [
            spaces.Box(low=0, high=self.num_agents - 1, shape=(self.num_agents * agent_id_dim_tuple[0],), dtype=np.int32)
        ] * self.num_agents
        self.edge_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=edge_dim_tuple, dtype=np.float32)] * self.num_agents


    def seed(self, seed=None):
        self.seed_val = seed # Store the seed
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return seed

    def reset(self)-> Tuple[arr, arr, arr, arr, arr, arr, List[Dict]]:
        """
        对环境进行重置, 返回初始观测数据
        Return:
        - obs:          观测数组
        - agent_id:     节点id数组
        - adj:        邻接矩阵数组
        - reward:     奖励数组
        - done:       完成布尔数组
        - info:       信息字典列表
        """

        self.current_steps = 0 # 重置步数计数为0

        # 对智能体进行初始随机化
        shuffled_agents = list(self.agents) 
        self.np_random.shuffle(shuffled_agents) 
        half = (self.num_agents + 1) // 2  
        cooperators = set(shuffled_agents[:half])                                                 
        for agent in self.agents:
            agent.position = self.np_random.random(2) * self.world_size 
            theta = self.np_random.random() * 2 * np.pi
            agent.direction_vector[0] = np.cos(theta)
            agent.direction_vector[1] = np.sin(theta)
            agent.current_payoff = np.array([0.0], dtype=np.float32)
            agent.strategy = np.array([1 if agent in cooperators else 0], dtype=np.int32)

        self._update_dist_adj() # 更新距离邻接矩阵

        # 执行观测
        obs, reward, adj, done, info = self._get_env_state()
        
        return obs, reward, adj, done, info
    
    
    def step(self, action_n: List) -> Tuple[arr, arr, arr, arr, arr, arr, List[Dict]]:
        """
        Return:
        - obs:          观测数组
        - agent_id:     节点id数组
        - adj:        邻接矩阵数组
        - reward:     奖励数组
        - done:       完成布尔数组
        - info:       信息字典列表

        """
        # 更新 step 计数
        self.current_steps += 1
        
        # 执行移动
        self._update_positions(action_n) # 更新位置和方向
        self._update_dist_adj()  # 更新距离邻接矩阵

        # 执行博弈
        payoffs = self._compute_payoffs() # 计算博弈收益
        self._record_payoffs(payoffs)  # 记录智能体当前收益
        self._update_strategies(payoffs) # Fermi规则更新智能体策略

        # 执行观测
        obs, reward, adj, done, info = self._get_env_state()  
        
        # print(f"[DEBUG]  obs: {obs.shape}, obs: {obs.dtype}")
        # print(f"[DEBUG]  rewards: {reward.shape}, rewards: {reward.dtype}")
        # print(f"[DEBUG]  adj: {adj.shape}, adj: {adj.dtype}")       

        return obs, reward, adj, done, info


    def _update_positions(self, action_n: List[arr]) -> None:
        """
        根据输入的动作列表, 更新所有智能体的速度方向和位置
        """
        if len(action_n) != self.num_agents:
            print(f"警告: _update_agent_positions 收到的动作列表长度 {len(action_n)} 与智能体数量 {self.num_agents} 不匹配。")
            return

        for i, agent in enumerate(self.agents): # 遍历智能体
            # 将动作转成 ndarray, float32 格式
            if not isinstance(action_n[i], np.ndarray):
                action = np.array(action_n[i], dtype=np.float32)
            else:
                action = action_n[i].astype(np.float32)            
            action = action.flatten()
            
            new_direction_vector = np.array([0.0, 0.0], dtype=np.float32) # 默认静止
            if action.shape[0] == 2:
                norm = np.linalg.norm(action)
                if norm > 1e-7: # 阈值可以根据需要调整
                    new_direction_vector = action / norm
            else:
                print(f"警告: Agent {agent.id} 在 _update_agent_positions 中收到形状无效的连续动作 {action} (期望形状 (2,)), 方向设为不动。")
            
            # 更新速度方向
            agent.direction_vector = new_direction_vector
            # 朝新方向运动固定长度更新位置 
            agent.position = (agent.position + agent.direction_vector * self.speed) % self.world_size


    def _update_dist_adj(self):
        """
        计算 self.dist_adj
        """
        positions = np.array([agent.position for agent in self.agents])
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2 
        self.dist_adj = np.linalg.norm(delta, axis=2)


    def _compute_payoffs(self) -> np.ndarray: 
        """
        Return:
        - payoffs: 智能体的收益数组, 形状 (N,)
        """
        N = self.num_agents
        payoffs = np.zeros(N, dtype=np.float32) 
        
        adj = self.dist_adj # 智能体间的距离矩阵       
        strategies = np.array([self.agents[idx].strategy[0] for idx in range(N)], dtype=np.int32) # 获取所有智能体当前的博弈策略 (C=1, D=0)

        # 遍历每个可能的博弈小组中心 i
        for i in range(N):
            group_members = np.where(adj[i] <= self.radius)[0] # 找到在 i 半径内的智能体的索引         
            num_group_members = len(group_members) # 索引长度即参与 i 发起的博弈的数目, 由于至少包括自身, 最小长度为1

            # 计算该小组内的合作者数量
            group_strategies = strategies[group_members] # 获取小组内智能体的策略
            num_group_cooperators = np.sum(group_strategies) # 小组内合作者个数
            
            # 计算小组公共池总收益和平均收益, 并累加
            total_group_payoff = num_group_cooperators * self.cost * self.r
            avg_group_payoff = total_group_payoff / num_group_members             
            payoffs[group_members] += avg_group_payoff # 将平均收益加到
            
            # 对于小组中的合作者，需要减去他们付出的成本            
            group_cooperators = group_members[group_strategies == 1] # 找到小组中的合作者索引
            if len(group_cooperators) > 0:
                payoffs[group_cooperators] -= self.cost
                
        return payoffs

    # # marl_env.py -> _compute_payoffs (临时修改用于诊断)
    # def _compute_payoffs(self) -> np.ndarray:
    #     N = self.num_agents
    #     payoffs = np.zeros(N, dtype=np.float32)
    #     adj = self.dist_adj
        
    #     # 简化任务：每个智能体的奖励等于它的邻居数量
    #     # 这个奖励是密集的、即时的、无欺骗性的
    #     for i in range(N):
    #         # 邻居数量（不包括自己）
    #         num_neighbors = np.sum((adj[i] > 0) & (adj[i] <= self.radius))
    #         payoffs[i] = float(num_neighbors)
            
    #     return payoffs


    def _record_payoffs(self, payoffs: np.ndarray) -> None:
        """ 更新 agent.current_payoff """
        for i, agent in enumerate(self.agents):
            agent.current_payoff = np.array([payoffs[i]], dtype=np.float32)


    def _update_strategies(self, payoffs: np.ndarray) -> None: # 参数类型改为 np.ndarray
        """ 更新 agent.strategy """
        N = self.num_agents
        adj = self.dist_adj

        next_strategies = [agent.strategy.copy() for agent in self.agents]         
        for i in range(N):
            neighbors = np.where((adj[i] > 0) & (adj[i] <= self.radius))[0] # 智能体 i 的所有邻居索引 (不包括自己，距离在 radius 内)
            if len(neighbors) == 0: # 如果没有邻居可以学习则策略不变
                continue
            j = self.np_random.choice(neighbors) # 随机选择一个邻居
            delta_payoff = payoffs[j] - payoffs[i]
            prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
            if self.np_random.random() < prob_adopt: # 以概率模仿
                next_strategies[i] = self.agents[j].strategy.copy()
        # 更新博弈策略
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]


    def _get_env_state(self):
        """
        获取每个step的环境状态
        Return:
        - obs: 所有智能体自身观测
        - reward_n: 所有智能体奖励
        - adj: 距离邻接矩阵
        - done: 环境终止信号
        - info: 环境信息字典
        """

        # 节点级state
        obs = [self._get_agent_obs(agent) for agent in self.agents] # 所有智能体自身观测列表
        reward = [self._get_agent_reward(agent) for agent in self.agents] # 所有智能体奖励列表

        # 环境级state
        adj = self.dist_adj.astype(np.float32) # 距离邻接矩阵
        num_cooperator = sum(1 for agent in self.agents if agent.strategy[0] == 1) # 合作者数量
        current_cooperation_rate = num_cooperator / self.num_agents if self.num_agents > 0 else 0.0 # 合作率
        current_total_reward = np.sum(np.concatenate(reward)) # 总奖励
        current_avg_reward = current_total_reward / self.num_agents if self.num_agents > 0 else 0.0 # 平均奖励

        # 环境信息
        done = (self.current_steps >= self.env_max_steps) # 终止信号
        info = {
                "step_cooperation_rate": current_cooperation_rate,
                "step_avg_reward": current_avg_reward,
                "current_steps": self.current_steps, 
            }
                

        return obs, reward, adj, done, info

    def _get_agent_obs(self, agent: Agent) -> arr:
        """
        Return:
        - obs: 由相对位置, 方向向量, 策略, 收益堆叠而成的6维数组, 是agent对自身状态的观测
        """    
        position = agent.position / self.world_size 
        direction_vector = agent.direction_vector 
        strategy = agent.strategy 
        current_payoff = agent.current_payoff
        agent_obs = np.hstack([
            position.flatten(),
            direction_vector.flatten(),
            strategy.astype(np.float32).flatten(), 
            current_payoff.flatten(),
        ]).astype(np.float32)

        return agent_obs

    
    def _get_agent_reward(self,  agent: Agent)  ->  arr:
        """
        Return:
        - reward: 智能体获得的奖励, 当前step的博弈收益
        """
        # 定义奖励为博弈的收益
        agent_reward = agent.current_payoff.copy()
        
        return agent_reward
