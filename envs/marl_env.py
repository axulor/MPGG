# marl_env.py

import gymnasium as gym
from gym import spaces
import numpy as np
from numpy import ndarray as arr
import argparse 
from typing import List, Tuple, Dict
import random
from gymnasium.utils import seeding
from . import visualize_utils 

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

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        # --- Basic Environment Parameters ---
        self.num_agents = args.num_agents
        self.world_size = args.world_size
        self.speed = args.speed
        
        # --- PGG Game Parameters (Now part of the Env) ---
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r # Synergy factor
        self.beta = args.beta # Selection intensity for Fermi rule
        self.k_neighbors = getattr(args, 'k_neighbors', 0) # k for k-NN interactions

        # --- Seeding ---
        self.seed_val = args.seed 
        self.np_random, _ = seeding.np_random(self.seed_val)
        random.seed(self.seed_val)

        # Initialize agents
        self.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.id = i
            agent.name = f"agent_{i}"

        # Environment state attributes
        self.current_steps = 0
        self.dist_adj = None # Distance adjacency matrix

        # Configure Gym spaces
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
        self.seed_val = seed
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return seed

    def reset(self, seed=None, options=None):
        # Allow seeding on reset
        if seed is not None:
            self.seed(seed)
            
        self.current_steps = 0

        # Initial random placement and strategy assignment
        shuffled_agents = list(self.agents) 
        self.np_random.shuffle(shuffled_agents) 
        half = (self.num_agents + 1) // 2  
        cooperators = set(shuffled_agents[:half])                                                 
        for agent in self.agents:
            agent.position = self.np_random.random(2) * self.world_size 
            theta = self.np_random.random() * 2 * np.pi
            agent.direction_vector = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            agent.current_payoff = np.array([0.0], dtype=np.float32)
            agent.strategy = np.array([1 if agent in cooperators else 0], dtype=np.int32)

        self._update_dist_adj()

        obs, reward, adj, done, info = self._get_env_state()
        
        # Gymnasium API expects obs and info as dicts for multi-agent envs, but let's stick to the project's list-based format for now.
        return obs, reward, adj, done, info
    
    def step(self, action_n: List) -> Tuple[arr, arr, arr, arr, List[Dict]]:
        """
        Executes one time step following the new "real game" logic.
        Flow: Move -> Play PGG -> Get Reward -> Evolve Strategy -> Observe
        """
        self.current_steps += 1
        
        # 1. Agents move based on RL policy actions
        self._update_positions(action_n)
        self._update_dist_adj()

        # 2. Play a single round of PGG to calculate payoffs
        # This function now computes the "average net payoff" for each agent.
        average_net_payoffs = self._calculate_pgg_payoffs()
        
        # 3. The calculated payoff is the agent's reward for the RL algorithm
        self._record_payoffs(average_net_payoffs)

        # 4. Agents evolve their strategies based on the payoffs from this step
        self._update_strategies(average_net_payoffs)

        # 5. Get the new environment state (with updated positions and possibly new strategies)
        obs, reward, adj, done, info = self._get_env_state()  
        
        return obs, reward, adj, done, info

    def _get_k_neighbor_mask(self, adj: np.ndarray) -> np.ndarray:
        """
        Calculates the k-nearest neighbor interaction mask, identical to the
        logic previously in pgg_sim.py.

        Returns a boolean mask (N, N), where mask[i, j] = True means agent j
        is one of i's k-nearest neighbors within the radius.
        """
        N = self.num_agents
        
        if self.k_neighbors <= 0:
            # Fallback to radius-based interaction if k is not specified
            return (adj > 0) & (adj <= self.radius)

        distances = adj.copy()
        # Exclude self and agents outside the radius
        mask = (adj == 0) | (adj > self.radius)
        distances[mask] = np.inf

        k = min(self.k_neighbors, N - 1)
        
        # Get indices of the k nearest neighbors for each agent
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

        # Create the boolean mask
        interaction_mask = np.zeros_like(adj, dtype=bool)
        rows = np.arange(N).repeat(k)
        cols = k_nearest_indices.flatten()
        
        # Only mark as True if the neighbor is actually within radius (not infinity)
        is_valid_neighbor = (distances[rows, cols] != np.inf)
        interaction_mask[rows[is_valid_neighbor], cols[is_valid_neighbor]] = True
        
        return interaction_mask
    
    def _calculate_pgg_payoffs(self) -> np.ndarray:
        """
        Vectorized calculation of each agent's "average net payoff" from a single
        round of the Public Goods Game. This function replaces the simulator.
        """
        N = self.num_agents
        strategies = np.array([agent.strategy[0] for agent in self.agents], dtype=np.float32)
        
        # 1. Determine who interacts with whom (directed k-NN graph)
        focal_game_mask = self._get_k_neighbor_mask(self.dist_adj)
        
        # 2. Define game groups: each agent 'i' initiates a game with itself and its neighbors
        focal_game_mask_with_self = focal_game_mask | np.eye(N, dtype=bool)
        
        # 3. Calculate payouts for each of the N initiated games
        game_group_sizes = focal_game_mask_with_self.sum(axis=1) # Shape (N,)
        game_num_cooperators = strategies @ focal_game_mask_with_self.T # Shape (N,)
        
        avg_pool_payouts = np.divide(
            game_num_cooperators * self.cost * self.r,
            game_group_sizes,
            out=np.zeros_like(game_num_cooperators, dtype=np.float32),
            where=(game_group_sizes > 0)
        )

        # 4. Calculate total gross gains for each agent
        # Agent j's gross gain is the sum of payouts from all games it participated in.
        total_gross_gains = avg_pool_payouts @ focal_game_mask_with_self.T # Shape (N,)

        # 5. Calculate total costs for each agent
        # Agent j's number of games is the count of how many times it appears in any group.
        n_games_played = focal_game_mask_with_self.T.sum(axis=0) # Shape (N,)
        total_costs = strategies * n_games_played * self.cost # Shape (N,)

        # 6. Calculate net gains and the final average payoff
        total_net_gains = total_gross_gains - total_costs
        average_net_payoffs = np.divide(
            total_net_gains,
            n_games_played,
            out=np.zeros_like(total_net_gains),
            where=(n_games_played > 0)
        )
        
        return average_net_payoffs
    
    def _update_strategies(self, payoffs: np.ndarray) -> None:
        """
        Synchronously updates all agents' strategies based on the Fermi rule,
        using the calculated payoffs from the current step. Each agent looks
        at one randomly chosen k-nearest neighbor to potentially imitate.
        """
        N = self.num_agents
        
        # Get the same interaction mask used for payoff calculation
        neighbor_mask = self._get_k_neighbor_mask(self.dist_adj)
        current_strategies = np.array([agent.strategy.copy() for agent in self.agents])
        next_strategies = current_strategies.copy()

        # Iterate through each agent to decide its next strategy
        for i in range(N):
            # Find agent i's potential imitation targets
            visible_neighbor_indices = np.where(neighbor_mask[i])[0]

            if len(visible_neighbor_indices) == 0:
                continue # No neighbors, no change in strategy

            # Randomly select one neighbor to compare with
            j = self.np_random.choice(visible_neighbor_indices)
            
            # Use Fermi rule to determine probability of adopting neighbor's strategy
            delta_payoff = payoffs[j] - payoffs[i]
            prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
            
            # Decide whether to adopt
            if self.np_random.random() < prob_adopt:
                next_strategies[i] = current_strategies[j]

        # Synchronously apply the new strategies to all agents
        for i, agent in enumerate(self.agents):
            agent.strategy = next_strategies[i]

    def _update_positions(self, action_n: List[arr]) -> None:
        """Updates agent positions based on actions (no changes from original)."""
        if len(action_n) != self.num_agents:
            print(f"Warning: action list length mismatch.")
            return

        for i, agent in enumerate(self.agents):
            action = np.array(action_n[i], dtype=np.float32).flatten()
            
            new_direction_vector = np.zeros(2, dtype=np.float32)
            if action.shape[0] == 2:
                norm = np.linalg.norm(action)
                if norm > 1e-7:
                    new_direction_vector = action / norm
            
            agent.direction_vector = new_direction_vector
            agent.position = (agent.position + agent.direction_vector * self.speed) % self.world_size

    def _update_dist_adj(self):
        """Updates the distance adjacency matrix (no changes from original)."""
        positions = np.array([agent.position for agent in self.agents])
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # Account for periodic boundary conditions (toroidal world)
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2 
        self.dist_adj = np.linalg.norm(delta, axis=2)

    def _record_payoffs(self, payoffs: np.ndarray) -> None:
        """Updates agent.current_payoff for observation and reward."""
        for i, agent in enumerate(self.agents):
            agent.current_payoff = np.array([payoffs[i]], dtype=np.float32)

    def _get_env_state(self):
        """Gathers observations, rewards, and info (no changes from original)."""
        obs = [self._get_agent_obs(agent) for agent in self.agents]
        reward = [self._get_agent_reward(agent) for agent in self.agents]
        adj = self.dist_adj.astype(np.float32)
        
        num_cooperators = sum(1 for agent in self.agents if agent.strategy[0] == 1)
        cooperation_rate = num_cooperators / self.num_agents if self.num_agents > 0 else 0.0
        avg_reward = np.mean([r[0] for r in reward])

        done = False # This environment does not have a terminal state
        info = {
            "step_cooperation_rate": cooperation_rate,
            "step_avg_reward": avg_reward,
            "current_steps": self.current_steps, 
        }
        return obs, reward, adj, done, info

    def _get_agent_obs(self, agent: Agent) -> arr:
        """Returns a single agent's observation vector (no changes from original)."""    
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

    def _get_agent_reward(self, agent: Agent) -> arr:
        """Returns the agent's reward, which is its calculated payoff."""
        return agent.current_payoff.copy()

    def render(self, mode: str = 'human'):
        """Renders the environment using visualize_utils."""
        render_data = self.get_render_data()
        rgb_array = visualize_utils.render_frame(
            render_data, self.world_size, self.radius, self.current_steps
        )
        if mode == 'rgb_array':
            return rgb_array
        elif mode == 'human':
            visualize_utils.display_frame(rgb_array)
            return None
        
    def close(self):
        """Closes the rendering window."""
        visualize_utils.close_render_window()

    def get_render_data(self) -> Dict[str, np.ndarray]:
        """Packs data for rendering."""
        return {
            "positions": np.array([agent.position for agent in self.agents]),
            "strategies": np.array([agent.strategy[0] for agent in self.agents]),
            "payoffs": np.array([agent.current_payoff[0] for agent in self.agents]),
            "adj": self.dist_adj,
        }