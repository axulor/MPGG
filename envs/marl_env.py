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

        # Parameters for the "Strategically-Guided Motion with Personal Space" model
        # This radius defines the "personal space" for the Separation rule.
        self.separation_radius = getattr(args, 'separation_radius', 0.125)


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
        edge_dim_tuple = (8,) 

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
        if seed is not None:
            self.seed(seed)
            
        self.current_steps = 0
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
        return obs, reward, adj, done, info
    
    def step(self, action_n: List) -> Tuple[arr, arr, arr, arr, List[Dict]]:
        self.current_steps += 1
        self._update_positions(action_n)
        self._update_dist_adj()
        average_net_payoffs = self._calculate_pgg_payoffs()
        self._record_payoffs(average_net_payoffs)
        self._update_strategies(average_net_payoffs)
        obs, reward, adj, done, info = self._get_env_state()  
        return obs, reward, adj, done, info

    def _get_k_neighbor_mask(self, adj: np.ndarray) -> np.ndarray:
        N = self.num_agents
        if self.k_neighbors <= 0:
            return (adj > 0) & (adj <= self.radius)
        distances = adj.copy()
        mask = (adj == 0) | (adj > self.radius)
        distances[mask] = np.inf
        k = min(self.k_neighbors, N - 1)
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
        interaction_mask = np.zeros_like(adj, dtype=bool)
        rows = np.arange(N).repeat(k)
        cols = k_nearest_indices.flatten()
        is_valid_neighbor = (distances[rows, cols] != np.inf)
        interaction_mask[rows[is_valid_neighbor], cols[is_valid_neighbor]] = True
        return interaction_mask
    
    # def _calculate_pgg_payoffs(self) -> np.ndarray:
    #     N = self.num_agents
    #     strategies = np.array([agent.strategy[0] for agent in self.agents], dtype=np.float32)
    #     focal_game_mask = self._get_k_neighbor_mask(self.dist_adj)
    #     focal_game_mask_with_self = focal_game_mask | np.eye(N, dtype=bool)
    #     game_group_sizes = focal_game_mask_with_self.sum(axis=1)
    #     game_num_cooperators = strategies @ focal_game_mask_with_self.T
    #     avg_pool_payouts = np.divide(
    #         game_num_cooperators * self.cost * self.r,
    #         game_group_sizes,
    #         out=np.zeros_like(game_num_cooperators, dtype=np.float32),
    #         where=(game_group_sizes > 0)
    #     )
    #     total_gross_gains = avg_pool_payouts @ focal_game_mask_with_self.T
    #     n_games_played = focal_game_mask_with_self.T.sum(axis=0)
    #     total_costs = strategies * n_games_played * self.cost
    #     total_net_gains = total_gross_gains - total_costs
    #     average_net_payoffs = np.divide(
    #         total_net_gains,
    #         n_games_played,
    #         out=np.zeros_like(total_net_gains),
    #         where=(n_games_played > 0)
    #     )
    #     return average_net_payoffs

    def _calculate_pgg_payoffs(self) -> np.ndarray:
        """
        Calculates payoffs based on a 'self-initiated game' model using vectorized operations.
        Each agent 'i' receives a payoff ONLY from the single game it initiates.
        The participants of this game are agent 'i' and its k-nearest neighbors.
        """
        N = self.num_agents
        strategies = np.array([agent.strategy[0] for agent in self.agents], dtype=np.float32)

        # 1. Determine the interaction groups for games initiated by each agent.
        #    focal_game_mask[i, j] = True means agent j is a neighbor of i.
        focal_game_mask = self._get_k_neighbor_mask(self.dist_adj)
        
        #    Add self to the group. `focal_game_mask_with_self[i, :]` are all participants
        #    in the game initiated by agent i.
        focal_game_mask_with_self = focal_game_mask | np.eye(N, dtype=bool)

        # 2. Calculate the size of each game group.
        #    game_group_sizes[i] is the number of participants in the game started by i.
        game_group_sizes = focal_game_mask_with_self.sum(axis=1) # Shape (N,)

        # 3. Calculate the number of cooperators in each initiated game.
        #    This is the sum of strategies of all participants for each game.
        #    game_num_cooperators[i] is the number of cooperators in game 'i'.
        game_num_cooperators = strategies @ focal_game_mask_with_self.T # Shape (N,)

        # 4. Calculate the gross per-capita payout FROM each initiated game.
        #    This is the amount each participant would get from the pool in the game initiated by agent 'i'.
        #    This vector represents the gross payoff for the initiator of each game.
        gross_payout_per_game = np.divide(
            game_num_cooperators * self.cost * self.r,
            game_group_sizes,
            out=np.zeros_like(game_num_cooperators, dtype=np.float32),
            where=(game_group_sizes > 0)
        )

        # 5. Calculate the net payoff for each agent.
        #    The cost for agent 'i' is simply `self.cost` if it's a cooperator, and 0 otherwise.
        #    It only pays this cost once for the game it initiates.
        costs = strategies * self.cost # Shape (N,)
        
        #    The net payoff for agent 'i' is the gross payout from its own game minus its own cost.
        #    This is a simple element-wise subtraction.
        net_payoffs = gross_payout_per_game - costs

        return net_payoffs
    
    # def _update_strategies(self, payoffs: np.ndarray) -> None:
    #     N = self.num_agents
    #     neighbor_mask = self._get_k_neighbor_mask(self.dist_adj)
    #     current_strategies = np.array([agent.strategy.copy() for agent in self.agents])
    #     next_strategies = current_strategies.copy()
    #     for i in range(N):
    #         visible_neighbor_indices = np.where(neighbor_mask[i])[0]
    #         if len(visible_neighbor_indices) == 0:
    #             continue
    #         j = self.np_random.choice(visible_neighbor_indices)
    #         delta_payoff = payoffs[j] - payoffs[i]
    #         prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
    #         if self.np_random.random() < prob_adopt:
    #             next_strategies[i] = current_strategies[j]
    #     for i, agent in enumerate(self.agents):
    #         agent.strategy = next_strategies[i]

    def _update_strategies(self, payoffs: np.ndarray) -> None:
        """
        Asynchronously updates one randomly chosen agent's strategy based on the
        Fermi rule. In each call to this function, only one agent gets the
        opportunity to update its strategy.
        """
        N = self.num_agents
        
        # 1. Randomly select ONE agent to potentially update its strategy.
        #    This is the core of asynchronous updating.
        focal_agent_id = self.np_random.integers(N)
        focal_agent = self.agents[focal_agent_id]

        # 2. Find the neighbors of this specific agent.
        neighbor_mask = self._get_k_neighbor_mask(self.dist_adj)
        visible_neighbor_indices = np.where(neighbor_mask[focal_agent_id])[0]

        # 3. If the agent has no neighbors, it cannot imitate anyone.
        if len(visible_neighbor_indices) == 0:
            return # No update occurs in this step.

        # 4. Randomly select ONE neighbor from its neighborhood to compare with.
        imitated_neighbor_id = self.np_random.choice(visible_neighbor_indices)
        imitated_neighbor = self.agents[imitated_neighbor_id]
        
        # 5. Apply the Fermi rule to decide whether to adopt the neighbor's strategy.
        #    The payoffs vector is from the current state of the environment.
        delta_payoff = payoffs[imitated_neighbor_id] - payoffs[focal_agent_id]
        prob_adopt = 1 / (1 + np.exp(-self.beta * delta_payoff))
        
        # 6. If the agent decides to adopt, update its strategy IMMEDIATELY.
        if self.np_random.random() < prob_adopt:
            focal_agent.strategy = imitated_neighbor.strategy.copy()


    def _update_positions(self, action_n: List[arr]) -> None:
        """
        Updates agent positions based on a hard physical embodiment constraint.
        This involves a two-stage process:
        1. Propose a move based on the RL policy.
        2. Detect and resolve any "collisions" (violations of personal space).
        """
        N = self.num_agents
        if N == 0:
            return
            
        positions = np.array([agent.position for agent in self.agents])

        # --- Stage 1: Propose New Positions based on RL Policy ---
        proposed_positions = np.zeros_like(positions)
        for i in range(N):
            agent = self.agents[i]
            rl_action = np.array(action_n[i], dtype=np.float32).flatten()
            rl_direction = self._normalize_vector(rl_action)
            
            # Update agent's direction based on RL policy
            agent.direction_vector = rl_direction
            
            # Calculate the proposed new position
            proposed_positions[i] = (positions[i] + rl_direction * self.speed) % self.world_size

        # --- Stage 2: Iteratively Resolve Collisions ---
        # We might need to loop a few times in case resolving one collision creates another.
        # 5 iterations is usually more than enough for convergence.
        for _ in range(5):
            collisions_found = False
            
            # Calculate all-pairs distances for the current positions (could be proposed or corrected)
            current_positions = proposed_positions
            delta = current_positions[:, np.newaxis, :] - current_positions[np.newaxis, :, :]
            delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2
            distances = np.linalg.norm(delta, axis=2)
            
            for i in range(N):
                # Find agents that are colliding with agent i
                colliding_mask = (distances[i] > 0) & (distances[i] < self.separation_radius)
                
                if np.any(colliding_mask):
                    collisions_found = True
                    
                    # For each collision, move both agents apart along the line connecting them.
                    for j in np.where(colliding_mask)[0]:
                        if i < j: # Process each pair only once to avoid double counting
                            dist = distances[i, j]
                            overlap = self.separation_radius - dist
                            
                            # The correction vector points from j to i
                            correction_vec = delta[i, j] / (dist + 1e-9)
                            
                            # Move both agents apart by half of the overlap
                            move_amount = overlap / 2.0
                            
                            # Apply correction and wrap around the world
                            proposed_positions[i] = (proposed_positions[i] + correction_vec * move_amount) % self.world_size
                            proposed_positions[j] = (proposed_positions[j] - correction_vec * move_amount) % self.world_size
                            
            if not collisions_found:
                break # If no collisions in a full pass, we are done.

        # --- Final Step: Apply the corrected positions to agents ---
        for i in range(N):
            self.agents[i].position = proposed_positions[i]

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        Helper function to normalize a 2D vector.
        Returns a zero vector if the norm is close to zero.
        """
        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            return vec / norm
        return np.zeros_like(vec, dtype=np.float32)

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
        # Add the repulsion radius to the render data for visualization
        render_data['repulsion_radius'] = self.repulsion_radius
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