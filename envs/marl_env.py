# --- START OF FILE marl_env.py ---

import gymnasium as gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
import argparse # 确保导入，即使 __init__ 中用的是 SimpleNamespace
from typing import List, Tuple, Dict
import random
from gymnasium.utils import seeding  # 用于环境随机种子的设置
from collections import deque 

# ==============================================================================
# == Agent Class Definition  ==
# ==============================================================================
class Agent:
    def __init__(self):
        self.id = None      # int
        self.name = None    # 字符串
        self.position = np.zeros(2, dtype=np.float32) # 位置
        self.direction_vector = np.zeros(2, dtype=np.float32) # 方向向量
        self.strategy = np.array([0], dtype=np.int32) # 0 for defector, 1 for cooperator
        self.current_payoff = np.array([0.0], dtype=np.float32) # 收益

# ==============================================================================
# == MultiAgentGraphEnv Class Definition  ==
# ==============================================================================
class MultiAgentGraphEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, args: argparse.Namespace): # 类型为 argparse.Namespace
        super().__init__()
        # --- 1. 参数初始化 ---
        self.num_agents = args.num_agents
        self.world_size = args.world_size
        self.speed = args.speed
        # self.episode_length = args.episode_length # 环境内部不需要知道runner的rollout segment长度
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r
        self.beta = args.beta
        self.env_max_steps = args.env_max_steps           # 环境最大步数限制
        self.cooperation_lower_threshold = args.cooperation_lower_threshold         # 最低合作率限制
        self.cooperation_upper_threshold = args.cooperation_upper_threshold         # 最高合作率限制
        self.sustain_duration = args.sustain_duration   # 合作率超出合理区间限制
        self.seed_val = args.seed # Store seed value for re-seeding if necessary
        self.seed(args.seed)


        # --- 2. 初始化智能体列表 ---
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # --- 3. 初始化环境状态 ---
        self.current_episode_steps = 0 # MODIFICATION: Renamed from current_step for clarity
        self.total_rewards_in_episode = np.zeros(self.num_agents, dtype=np.float32) # MODIFICATION: Tracks rewards within a LOGICAL episode
        self.cooperation_counts_in_episode = np.zeros(self.num_agents, dtype=np.int32) # MODIFICATION: Tracks coop counts within a LOGICAL episode
        self.cooperation_rate_deque = deque(maxlen=self.sustain_duration) 

        # --- 4. 图和距离属性 ---
        self.edge_list = None
        self.edge_weight = None
        self.cached_dist_mag = None

        # --- 5. 配置 Gym 空间  ---
        self.agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = [self.agent_action_space] * self.num_agents

        temp_agent = Agent() 
        temp_agent.position = np.zeros(2); temp_agent.direction_vector = np.zeros(2)
        temp_agent.strategy = np.array([0]); temp_agent.current_payoff = np.array([0.0])
        obs_sample = self.get_agent_feat(temp_agent) 
        obs_dim = obs_sample.shape[0]

        self.agent_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = [self.agent_observation_space] * self.num_agents

        share_obs_dim = obs_dim * self.num_agents
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.num_agents

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

    def reset(self) -> Tuple[List[arr], List[arr], List[arr], List[arr]]:
        self.cooperation_rate_deque.clear() 
        self.total_rewards_in_episode.fill(0.0) # MODIFICATION: Reset episode-specific counters
        self.cooperation_counts_in_episode.fill(0) # MODIFICATION: Reset episode-specific counters
        self.current_episode_steps = 0 # MODIFICATION: Reset episode step counter

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

        self.calculate_distances()
        self.update_graph()

        obs_n = [self.get_obs(agent) for agent in self.agents]
        agent_id_n = [self.get_id(agent) for agent in self.agents]
        node_obs_n_all, adj_n_all = self.get_graph_obs()
        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents
        
        return obs_n, agent_id_n, node_obs_n, adj_n
    
    def check_cooperation_rate(self, current_cooperation_rate: float) -> bool:
        self.cooperation_rate_deque.append(current_cooperation_rate)
        if len(self.cooperation_rate_deque) < self.sustain_duration:
            return False
        all_below_lower = all(rate < self.cooperation_lower_threshold for rate in self.cooperation_rate_deque)
        all_above_upper = all(rate > self.cooperation_upper_threshold for rate in self.cooperation_rate_deque)
        return all_below_lower or all_above_upper
    
    def step(self, action_n: List) -> Tuple[List[arr], List[arr], List[arr], List[arr], List[arr], List[bool], List[Dict]]:
        self.current_episode_steps += 1 # MODIFICATION: Increment episode step counter

        for i, agent in enumerate(self.agents):
            self.update_direction_vector(action_n[i], agent)
            agent.position = (agent.position + agent.direction_vector * self.speed) % self.world_size

        self.calculate_distances()
        self.update_graph()

        payoffs = self.compute_payoffs()
        self.record_payoffs(payoffs) 
        self.update_strategies(payoffs) 

        obs_n = [self.get_obs(agent) for agent in self.agents]
        agent_id_n = [self.get_id(agent) for agent in self.agents]
        node_obs_n_all, adj_n_all = self.get_graph_obs()
        node_obs_n = [node_obs_n_all] * self.num_agents
        adj_n = [adj_n_all] * self.num_agents

        num_cooperator = sum(1 for agent in self.agents if agent.strategy[0] == 1)
        current_cooperation_rate = num_cooperator / self.num_agents if self.num_agents > 0 else 0.0
        current_total_reward = sum(agent.current_payoff[0] for agent in self.agents)
        current_avg_reward = current_total_reward / self.num_agents if self.num_agents > 0 else 0.0

        current_step_payoffs = np.array([agent.current_payoff[0] for agent in self.agents], dtype=np.float32)
        for i, agent in enumerate(self.agents):
            self.total_rewards_in_episode[i] += current_step_payoffs[i] # MODIFICATION: Accumulate for current LOGICAL episode
            if agent.strategy[0] == 1:
                self.cooperation_counts_in_episode[i] += 1 # MODIFICATION: Accumulate for current LOGICAL episode

        terminate_by_rate = self.check_cooperation_rate(current_cooperation_rate)
        terminate_by_timeout = (self.current_episode_steps >= self.env_max_steps)
        
        is_logic_done = terminate_by_rate or terminate_by_timeout # This is the "logical" done for an episode
        
        # MODIFICATION START: Add is_absorb_state to info
        is_absorb_state = False
        if is_logic_done: # Only check for absorb state if the episode is logically done
            if terminate_by_rate: # if done by rate, it's an absorb state by definition here
                 is_absorb_state = True
            # Could add more sophisticated checks for absorb state if needed,
            # e.g. if cooperation rate is 0 or 1 for X steps even if not hitting sustain_duration for *extreme* rates
        # MODIFICATION END

        reward_n = []
        done_n = [] # This will be the logical done signal for each agent
        info_n = []

        for i in range(self.num_agents):
            reward_n.append(np.array([current_step_payoffs[i]], dtype=np.float32))
            done_n.append(is_logic_done) 

            agent_i_info = {
                "step_cooperation_rate": current_cooperation_rate,
                "step_avg_reward": current_avg_reward,
                "bad_mask_indicator": True, # Always True for GAE in non-terminating (or pseudo-terminating) envs
                "is_absorb_state": is_absorb_state, # MODIFICATION: Added flag
                "current_episode_steps": self.current_episode_steps, # MODIFICATION: Added current logical episode steps
                # Episode statistics are added if is_logic_done is True
                "episode_total_reward_agent": self.total_rewards_in_episode[i] if is_logic_done else None,
                "episode_coop_count_agent": self.cooperation_counts_in_episode[i] if is_logic_done else None,
                "episode_length": self.current_episode_steps if is_logic_done else None,
            }
            # If the episode is logically done, we might want to add aggregated episode stats to agent 0's info for logging
            if i == 0 and is_logic_done:
                agent_i_info["episode_mean_cooperation_rate"] = np.sum(self.cooperation_counts_in_episode) / (self.num_agents * self.current_episode_steps) if self.num_agents * self.current_episode_steps > 0 else 0
                agent_i_info["episode_total_social_reward"] = np.sum(self.total_rewards_in_episode)

            info_n.append(agent_i_info)
        
        # If a logical episode is done, the internal state for the *next* call to step()
        # (if the runner decides to continue without reset) should reflect a new logical episode.
        # However, the actual reset of counters (like self.current_episode_steps) happens in self.reset().
        # The VecEnv wrapper, if not resetting, will just continue calling step on this env instance.
        # So, if is_logic_done is True, the *next* step taken by this env instance (if not reset by VecEnv)
        # will effectively be step 1 of a new logical episode.
        # This means the episode counters should be reset here if we want them to be accurate for a *new* logical episode
        # that starts *without* an explicit env.reset() call from the Runner.
        # This is tricky. Standard gym envs reset all state in env.reset().
        # If the Runner doesn't call reset, then these counters will just keep accumulating across logical episodes
        # until an explicit reset. This is usually fine, as the Runner gets the 'episode_length' from info
        # when 'is_logic_done' is true, and can use that.
        # For now, let's assume Runner uses the info at 'is_logic_done' and then Buffer handles segments.
        # The counters (self.total_rewards_in_episode, etc.) will be reset only upon an explicit env.reset().

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

    def calculate_distances(self):
        positions = np.array([agent.position for agent in self.agents])
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2 
        self.cached_dist_mag = np.linalg.norm(delta, axis=2) 

    def compute_payoffs(self) -> Dict[int, float]:
        N = self.num_agents
        payoffs = {i: 0.0 for i in range(N)}
        dmat = self.cached_dist_mag
        for i in range(N):
            group_indices_for_game_i = np.where(dmat[i] <= self.radius)[0] 
            if len(group_indices_for_game_i) == 0:
                continue 
            total_contribution_in_group_i = 0
            for member_idx in group_indices_for_game_i:
                if self.agents[member_idx].strategy[0] == 1: 
                    total_contribution_in_group_i += self.cost
            pool_amount_from_group_i = total_contribution_in_group_i * self.r
            if len(group_indices_for_game_i) > 0:
                share_per_member_from_group_i = pool_amount_from_group_i / len(group_indices_for_game_i)
            else:
                share_per_member_from_group_i = 0.0
            if share_per_member_from_group_i > 0 or total_contribution_in_group_i > 0: 
                for member_idx in group_indices_for_game_i:
                    payoff_from_this_game = share_per_member_from_group_i 
                    if self.agents[member_idx].strategy[0] == 1:
                        payoff_from_this_game -= self.cost
                    payoffs[member_idx] += payoff_from_this_game
        return payoffs

    def update_strategies(self, payoffs: Dict[int, float]):
        N = self.num_agents
        dmat = self.cached_dist_mag
        next_strategies = [agent.strategy.copy() for agent in self.agents]
        current_payoffs_arr = np.array([payoffs[i] for i in range(N)]) 
        for i in range(N):
            neighbor_indices = np.where((dmat[i] > 0) & (dmat[i] <= self.radius))[0]
            if len(neighbor_indices) == 0: 
                continue
            j = self.np_random.choice(neighbor_indices)
            delta_payoff = current_payoffs_arr[j] - current_payoffs_arr[i]
            prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
            if self.np_random.random() < prob_adopt:
                next_strategies[i] = self.agents[j].strategy.copy()
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]

    def record_payoffs(self, payoffs: Dict[int, float]):
        for i, ai in enumerate(self.agents):
            ai.current_payoff = np.array([payoffs[i]], dtype=np.float32)

    def update_graph(self):
        if self.cached_dist_mag is None: self.calculate_distances()
        dists = self.cached_dist_mag
        connect_mask = (dists > 0) & (dists <= self.radius)
        row, col = np.where(connect_mask)
        self.edge_list = np.stack([row, col]) 
        self.edge_weight = dists[row, col]

    def get_agent_feat(self, agent: Agent) -> arr:       
        position = agent.position / self.world_size 
        direction_vector = agent.direction_vector 
        strategy = agent.strategy 
        current_payoff = agent.current_payoff
        features = np.hstack([
            position.flatten(),
            direction_vector.flatten(),
            strategy.astype(np.float32).flatten(), 
            current_payoff.flatten(),
        ]).astype(np.float32)
        return features

    def get_obs(self, agent: Agent) -> arr:
        return self.get_agent_feat(agent)

    def get_reward(self, agent: Agent) -> np.ndarray:
        return np.array([float(agent.current_payoff[0])], dtype=np.float32)

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.id], dtype=np.int32)

    def get_graph_obs(self) -> Tuple[arr, arr]:
        node_features = [self.get_agent_feat(agent) for agent in self.agents]
        node_obs = np.array(node_features, dtype=np.float32)
        adj = self.cached_dist_mag.astype(np.float32)
        return node_obs, adj

    def update_direction_vector(self, action: arr, agent: Agent) -> None:
        if not isinstance(action, np.ndarray):
            action_vec = np.array(action, dtype=np.float32)
        else:
            action_vec = action.astype(np.float32)
        action_vec = action_vec.flatten()
        if action_vec.shape[0] == 2:
            norm = np.linalg.norm(action_vec)
            if norm > 1e-7: 
                agent.direction_vector = action_vec / norm
            else:
                agent.direction_vector = np.array([0.0, 0.0], dtype=np.float32)
        else:
            print(f"警告: Agent {agent.id} 的 update_direction_vector 收到形状无效的连续动作 {action_vec} (期望形状 (2,)), 方向设为不动。")
            agent.direction_vector = np.array([0.0, 0.0], dtype=np.float32)