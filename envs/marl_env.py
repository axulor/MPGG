import gymnasium as gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
import argparse 
from typing import List, Tuple, Dict
import random
from gymnasium.utils import seeding
from . import visualize_utils 
from .pgg_sim import PGGSimulator  

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

        self.seed_val = args.seed 
        self.seed(args.seed)

        # 从 args 获取新的模拟器参数
        self.egt_rounds = args.egt_rounds
        self.egt_steps = args.egt_steps


        # 初始化智能体列表
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # 环境自身属性
        self.current_steps = 0 # 标记执行的步数计数
        self.dist_adj = None # 距离邻接矩阵

        # [新增] 初始化 PGG 模拟器
        self.simulator = PGGSimulator(args)
        

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
        edge_dim_tuple = (7,) 

        self.node_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=node_obs_dim_tuple, dtype=np.float32)] * self.num_agents
        self.adj_observation_space = [spaces.Box(low=0, high=+np.inf, shape=adj_dim_tuple, dtype=np.float32)] * self.num_agents
        self.agent_id_observation_space = [spaces.Box(low=0, high=self.num_agents - 1, shape=agent_id_dim_tuple, dtype=np.int32)] * self.num_agents
        self.share_agent_id_observation_space = [
            spaces.Box(low=0, high=self.num_agents - 1, shape=(self.num_agents * agent_id_dim_tuple[0],), dtype=np.int32)
        ] * self.num_agents
        self.edge_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=edge_dim_tuple, dtype=np.float32)] * self.num_agents

        self.seed(args.seed)

    def seed(self, seed=None):
        self.seed_val = seed # Store the seed
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        # [新增] 为模拟器设置种子，确保模拟过程的可复现性
        if hasattr(self, 'simulator'):
            self.simulator.seed(seed)
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
    
    
    # def step(self, action_n: List) -> Tuple[arr, arr, arr, arr, arr, arr, List[Dict]]:
    #     """
    #     [重构后的 Step 函数]
    #     执行移动 -> 调用模拟器评估 -> 获取状态
    #     """
    #     self.current_steps += 1
        
    #     # 1. 智能体根据RL策略移动 (慢时间尺度)
    #     self._update_positions(action_n)
    #     self._update_dist_adj()

    #     # 2. 准备并运行内部演化博弈模拟 (快时间尺度，虚拟评估)
    #     #    提取所有智能体的固定策略（身份）
    #     initial_strategies = np.array([agent.strategy[0] for agent in self.agents])
        
    #     #    调用模拟器进行虚拟评估，它返回对当前空间位置的适应性得分
    #     evolutionary_scores = self.simulator.run_simulation(initial_strategies, self.dist_adj)
        
    #     # 3. 将模拟得到的适应性得分作为本宏观步骤的奖励
    #     #    注意：模拟器不再改变 agent.strategy
    #     self._record_payoffs(evolutionary_scores)

    #     # 4. 获取环境状态并返回给RL算法
    #     obs, reward, adj, done, info = self._get_env_state()  
        
    #     return obs, reward, adj, done, info

    def step(self, action_n: List) :
        """
        [混合演化模型的 Step 函数]
        RL移动 -> 模拟器评估收益 -> 真实策略演化 -> 获取环境状态
        """
        self.current_steps += 1
        
        # 1. 智能体根据RL策略移动 (慢时间尺度)
        self._update_positions(action_n)
        self._update_dist_adj()

        # 2. 调用模拟器进行“虚拟评估”以计算本轮奖励
        #    输入的是当前“真实”的策略和位置
        current_real_strategies = np.array([agent.strategy[0] for agent in self.agents])
        
        #    模拟器运行，但它的目的【仅仅是】为了计算出智能体在当前位置的
        #    “演化适应性得分” (Evolutionary Fitness Score)。
        #    这个得分将作为RL的奖励。
        #    我们把这个返回值命名为 reward_payoffs 以明确其用途。
        reward_payoffs = self.simulator.run_simulation(current_real_strategies, self.dist_adj)
        
        # 3. 将模拟得到的适应性得分作为本宏观步骤的【奖励】
        self._record_payoffs(reward_payoffs) # agent.current_payoff 更新，用于计算reward

        # 4. 【关键新增步骤】执行一次“真实策略演化”
        #    使用刚刚计算出的、代表长期潜力的 reward_payoffs，
        #    在【真实环境】中根据Fermi规则进行【一次】策略更新。
        #    这会直接修改 self.agents 中智能体的 agent.strategy 属性。
        self._update_strategies(reward_payoffs) # 这个函数现在被重新启用了！

        # 5. 获取环境状态并返回给RL算法
        #    返回的 obs 将包含可能已经改变了的新策略。
        #    奖励是基于移动前的策略、在移动后的位置上评估得出的。
        obs, reward, adj, done, info = self._get_env_state()  
        
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


    # def _compute_payoffs(self) -> np.ndarray: 
    #     """
    #     [MODIFIED] 计算每个智能体的 *平均* 博弈收益。
    #     """
    #     N = self.num_agents
    #     # payoffs 数组现在用来累加总收益
    #     payoffs = np.zeros(N, dtype=np.float32)
    #     # [NEW] n_games_played 数组用来记录每个智能体参与了多少次博弈
    #     n_games_played = np.zeros(N, dtype=np.int32)
        
    #     adj = self.dist_adj
    #     strategies = np.array([self.agents[idx].strategy[0] for idx in range(N)])

    #     # --- 第一阶段：累加每个小组带来的收益 ---
    #     for i in range(N): # 遍历由智能体 i 发起（或为中心）的博弈小组
    #         group_members_indices = np.where(adj[i] <= self.radius)[0]
            
    #         # 记录这个小组的所有成员都多参与了一次博弈
    #         n_games_played[group_members_indices] += 1
            
    #         num_group_members = len(group_members_indices)
    #         if num_group_members == 0: continue

    #         group_strategies = strategies[group_members_indices]
    #         num_group_cooperators = np.sum(group_strategies)
            
    #         # 计算这个小组的公共池平均收益
    #         total_group_payoff = num_group_cooperators * self.cost * self.r
    #         avg_group_payoff = total_group_payoff / num_group_members
            
    #         # 将这个小组的平均收益，累加到所有小组成员的总收益上
    #         payoffs[group_members_indices] += avg_group_payoff
            
    #         # 对于小组中的合作者，他们需要付出成本
    #         # 找到这个小组中的合作者，在他们的总收益中减去成本
    #         # (注意：是 -=c，而不是 +=(-c))
    #         coop_in_group_mask = (group_strategies == 1)
    #         cooperator_indices_in_group = group_members_indices[coop_in_group_mask]
    #         if len(cooperator_indices_in_group) > 0:
    #             payoffs[cooperator_indices_in_group] -= self.cost
                
    #     # --- 第二阶段：计算平均收益 ---
    #     # 避免除以零的错误
    #     # 如果一个智能体没有参与任何博弈 (n_games_played[i] == 0)，它的平均收益就是0。
    #     # np.divide 的 where 参数可以很方便地处理这种情况。
    #     average_payoffs = np.divide(payoffs, n_games_played, 
    #                                 out=np.zeros_like(payoffs), 
    #                                 where=(n_games_played != 0))
                
    #     return average_payoffs


    # def _compute_payoffs(self) -> np.ndarray: 
    #     """
    #     Return:
    #     - payoffs: 智能体的收益数组, 形状 (N,)
    #     """
    #     N = self.num_agents
    #     payoffs = np.zeros(N, dtype=np.float32) 
        
    #     adj = self.dist_adj # 智能体间的距离矩阵       
    #     strategies = np.array([self.agents[idx].strategy[0] for idx in range(N)], dtype=np.int32) # 获取所有智能体当前的博弈策略 (C=1, D=0)

    #     # 遍历每个可能的博弈小组中心 i
    #     for i in range(N):
    #         group_members = np.where(adj[i] <= self.radius)[0] # 找到在 i 半径内的智能体的索引         
    #         num_group_members = len(group_members) # 索引长度即参与 i 发起的博弈的数目, 由于至少包括自身, 最小长度为1

    #         # 计算该小组内的合作者数量
    #         group_strategies = strategies[group_members] # 获取小组内智能体的策略
    #         num_group_cooperators = np.sum(group_strategies) # 小组内合作者个数
            
    #         # 计算小组公共池总收益和平均收益, 并累加
    #         total_group_payoff = num_group_cooperators * self.cost * self.r
    #         avg_group_payoff = total_group_payoff / num_group_members             
    #         payoffs[group_members] += avg_group_payoff # 将平均收益加到
            
    #         # 对于小组中的合作者，需要减去他们付出的成本            
    #         group_cooperators = group_members[group_strategies == 1] # 找到小组中的合作者索引
    #         if len(group_cooperators) > 0:
    #             payoffs[group_cooperators] -= self.cost
                
    #     return payoffs

    def _record_payoffs(self, payoffs: np.ndarray) -> None:
        """ 更新 agent.current_payoff """
        for i, agent in enumerate(self.agents):
            agent.current_payoff = np.array([payoffs[i]], dtype=np.float32)

    def _update_strategies(self, payoffs: np.ndarray) -> None:
        """
        [MODIFIED] 根据模拟器返回的长期适应性收益，同步更新所有智能体的真实策略。

        此函数执行以下操作：
        1. 对输入的收益数组 `payoffs` 进行归一化，使其范围在 [0, 1] 之间。
        2. 对每个智能体 i，从其半径范围内的 k 个最近邻居中，随机选择一个邻居 j。
        3. 使用归一化后的收益差，通过 Fermi 规则计算模仿概率。
        4. 以该概率，智能体 i 在下一步骤将采纳智能体 j 的策略。
        5. 所有智能体的策略是同步更新的。
        """
        N = self.num_agents
        adj = self.dist_adj
        
        # 1. 【关键修改】对输入的收益进行归一化
        min_p = np.min(payoffs)
        max_p = np.max(payoffs)
        
        normalized_payoffs = np.full_like(payoffs, 0.5) # 默认值
        if max_p - min_p > 1e-7: # 避免除以零
            normalized_payoffs = (payoffs - min_p) / (max_p - min_p)

        # 准备一个数组来存储下一轮的策略
        next_strategies = [agent.strategy.copy() for agent in self.agents]
        
        # 对每个智能体进行策略更新决策
        for i in range(N):
            # 2. 【关键修改】选择可见邻居 (k-nearest within radius)
            
            # 获取智能体 i 的所有距离数据
            distances_i = adj[i].copy()
            
            # 排除自身和超出半径的个体，将它们的距离设为无穷大
            distances_i[i] = np.inf 
            distances_i[distances_i > self.radius] = np.inf
            
            # 对距离进行排序，得到邻居的索引
            sorted_neighbor_indices = np.argsort(distances_i)
            
            # 从排序后的索引中，选出所有有效的邻居（距离不是无穷大）
            valid_neighbors = sorted_neighbor_indices[distances_i[sorted_neighbor_indices] != np.inf]
            
            # 从有效邻居中，最多选择 k 个
            # getattr(self, 'k_neighbors', N) 会尝试获取 k_neighbors，如果不存在则默认为所有邻居
            k = getattr(self, 'k_neighbors', N) 
            visible_neighbors = valid_neighbors[:k]

            # 如果没有可见的邻居，则不更新策略
            if len(visible_neighbors) == 0:
                continue
                
            # 3. 从可见邻居中随机选择一个模仿对象
            j = self.np_random.choice(visible_neighbors)
            
            # 4. 使用【归一化后】的收益计算 Fermi 规则
            delta_payoff = normalized_payoffs[j] - normalized_payoffs[i]
            
            # 为避免 exp 溢出，可以对输入进行裁剪 (可选但推荐)
            exp_arg = np.clip(-self.beta * delta_payoff, -700, 700)
            prob_adopt = 1 / (1 + np.exp(exp_arg))
            
            # 5. 以计算出的概率进行模仿
            if self.np_random.random() < prob_adopt:
                next_strategies[i] = self.agents[j].strategy.copy()

        # 6. 所有决策完成后，同步更新所有智能体的策略
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]

    # def _update_strategies(self, payoffs: np.ndarray) -> None: # 参数类型改为 np.ndarray
    #     """ 同步更新 agent.strategy """
    #     N = self.num_agents
    #     adj = self.dist_adj

    #     next_strategies = [agent.strategy.copy() for agent in self.agents]         
    #     for i in range(N):
    #         neighbors = np.where((adj[i] > 0) & (adj[i] <= self.radius))[0] # 智能体 i 的所有邻居索引 (不包括自己，距离在 radius 内)
    #         if len(neighbors) == 0: # 如果没有邻居可以学习则策略不变
    #             continue
    #         j = self.np_random.choice(neighbors) # 随机选择一个邻居
    #         delta_payoff = payoffs[j] - payoffs[i]
    #         prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
    #         if self.np_random.random() < prob_adopt: # 以概率模仿
    #             next_strategies[i] = self.agents[j].strategy.copy()
    #     # 更新博弈策略
    #     for i, agent in enumerate(self.agents):
    #         agent.strategy = next_strategies[i]

    # def _update_strategies(self, payoffs: np.ndarray) -> None:
    #     """
    #     Asynchronously updates agent strategies using the Fermi rule.
    #     In each call, only ONE randomly chosen agent gets to update its strategy.
    #     """
    #     N = self.num_agents
    #     adj = self.dist_adj

    #     # 1. 随机选择一个智能体进行更新
    #     agent_to_update_idx = self.np_random.integers(N)
    #     agent_to_update = self.agents[agent_to_update_idx]

    #     # 2. 该智能体随机选择一个邻居进行模仿
    #     neighbors = np.where((adj[agent_to_update_idx] > 0) & (adj[agent_to_update_idx] <= self.radius))[0]
        
    #     # 如果没有邻居，则不进行更新
    #     if len(neighbors) == 0:
    #         return

    #     # 随机选择一个邻居作为模仿对象
    #     imitation_target_idx = self.np_random.choice(neighbors)
        
    #     # 3. 执行Fermi规则
    #     delta_payoff = payoffs[imitation_target_idx] - payoffs[agent_to_update_idx]
        
    #     # 使用裁剪来避免计算 exp 时溢出
    #     exp_input = np.clip(-self.beta * delta_payoff, -50, 50)
    #     adoption_prob = 1 / (1 + np.exp(exp_input))
        
    #     # 4. 以计算出的概率，更新该智能体的策略
    #     if self.np_random.random() < adoption_prob:
    #         # 直接修改 agent 对象的策略
    #         target_strategy = self.agents[imitation_target_idx].strategy.copy()
    #         agent_to_update.strategy = target_strategy


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
        done = False # 终止信号
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
    

    def render(self, mode: str = 'human'):
        """
        Renders the environment.
        - 'human' mode: Tries to display a window on screen. Not recommended for multi-processing.
        - 'rgb_array' mode: Returns an RGB numpy array, suitable for saving to a file or GIF.
        """
        # 1. 获取所有渲染所需的数据
        render_data = self.get_render_data()

        # 2. 调用外部工具函数来生成图像帧
        # 这个函数总是返回一个RGB数组
        rgb_array = visualize_utils.render_frame(
            render_data,
            self.world_size,
            self.radius,
            self.current_steps
        )

        # 3. 根据模式决定如何处理图像帧
        if mode == 'rgb_array':
            return rgb_array
        elif mode == 'human':
            # 在 human 模式下，我们尝试显示这个图像
            # 需要在 visualize_utils 中增加一个显示函数
            visualize_utils.display_frame(rgb_array)
            return None # human 模式不返回值
        else:
            super(MultiAgentGraphEnv, self).render(mode=mode) # 调用父类处理不支持的模式
        
    def close(self):
        """
        Closes any open resources by calling the external utility.
        """
        visualize_utils.close_render_window()

    def get_render_data(self) -> Dict[str, np.ndarray]:
        """
        Packs all necessary information for rendering into a dictionary.
        """
        return {
            "positions": np.array([agent.position for agent in self.agents]),
            "strategies": np.array([agent.strategy[0] for agent in self.agents]),
            "payoffs": np.array([agent.current_payoff[0] for agent in self.agents]),
            "adj": self.dist_adj # [ADDED] Ensure adj is provided
        }
    
    def set_state_from_obs_and_adj(self, obs: np.ndarray, adj: np.ndarray):
        """
        [NEW HELPER] Manually sets the internal state of the environment
        based on observation and adjacency data. Used for post-hoc rendering.
        
        Args:
            obs (np.ndarray): Shape (N, D_obs)
            adj (np.ndarray): Shape (N, N)
        """
        if obs.shape[0] != self.num_agents:
            raise ValueError("Observation array has incorrect number of agents.")
        
        for i, agent in enumerate(self.agents):
            agent_obs = obs[i]
            # Unpack the 6D observation vector back into agent attributes
            # obs: (pos(2), dir(2), strat(1), payoff(1))
            # Note: We are primarily interested in position and strategy for rendering.
            
            # We cannot fully recover original position from normalized position,
            # but for rendering it's fine to use the scaled one.
            agent.position = agent_obs[0:2] * self.world_size
            agent.direction_vector = agent_obs[2:4]
            agent.strategy[0] = int(agent_obs[4])
            agent.current_payoff[0] = agent_obs[5]
            
        self.dist_adj = adj.copy()
