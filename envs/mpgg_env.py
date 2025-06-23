import gymnasium as gym
from gym import spaces
import numpy as np
from gymnasium.utils import seeding

class VectorizedMultiAgentEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_agents, world_size, speed, radius, cost, r, beta, max_steps, seed=None):
        super().__init__()
        # store parameters
        self.N = num_agents
        self.world_size = world_size
        self.speed = speed
        self.radius = radius
        self.cost = cost
        self.r = r
        self.beta = beta
        self.max_steps = max_steps
        # rng
        self.seed_val = seed
        self.seed(seed)
        # internal state arrays
        self.positions = np.zeros((self.N, 2), dtype=np.float32)
        self.directions = np.zeros((self.N, 2), dtype=np.float32)
        self.strategies = np.zeros((self.N,), dtype=np.int32)
        self.payoffs = np.zeros((self.N,), dtype=np.float32)
        self.current_steps = 0
        # observation dims
        obs_dim = 2 + 2 + 1 + 1  # pos, dir, strat, payoff
        # define spaces
        self.action_space = [spaces.Box(-1.0, 1.0, (2,), np.float32)] * self.N
        self.observation_space = [spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)] * self.N
        share_obs_dim = obs_dim * self.N
        self.share_observation_space = [
            spaces.Box(-np.inf, np.inf, (share_obs_dim,), np.float32)
        ] * self.N
        self.node_observation_space = [
            spaces.Box(-np.inf, np.inf, (self.N, obs_dim), np.float32)
        ] * self.N
        self.adj_observation_space = [
            spaces.Box(0.0, np.inf, (self.N, self.N), np.float32)
        ] * self.N

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def reset(self):
        # reset step count
        self.current_steps = 0
        # randomize positions & directions
        self.positions = self.np_random.random((self.N, 2)) * self.world_size
        angles = self.np_random.random(self.N) * 2 * np.pi
        self.directions[:, 0] = np.cos(angles)
        self.directions[:, 1] = np.sin(angles)
        # half cooperators
        idx = np.arange(self.N)
        self.np_random.shuffle(idx)
        half = (self.N + 1) // 2
        self.strategies.fill(0)
        self.strategies[idx[:half]] = 1
        self.payoffs.fill(0.0)
        # adjacency
        self._update_adj()
        # build observations
        return self._get_state()

    def step(self, actions):
        # actions: list or array (N,2)
        self.current_steps += 1
        # update directions & positions vectorized
        acts = np.vstack(actions).astype(np.float32)
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        mask = norms > 1e-7
        self.directions = np.where(mask, acts / norms, 0.0)
        self.positions = (self.positions + self.directions * self.speed) % self.world_size
        # adjacency
        self._update_adj()
        # payoff
        self._compute_payoffs()
        # strategy update (vectorized sampling)
        self._update_strategies()
        # get next state
        return self._get_state()

    def _update_adj(self):
        # toroidal distances
        delta = self.positions[:, None, :] - self.positions[None, :, :]
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2
        self.dist_mat = np.linalg.norm(delta, axis=-1)

    def _compute_payoffs(self):
        # mask of neighbors including self
        G = (self.dist_mat <= self.radius).astype(np.float32)
        # count cooperators per group
        coop_count = G @ self.strategies
        group_size = G.sum(axis=1)
        avg_pay = coop_count * self.cost * self.r / group_size
        # accumulate payoffs
        pay = G.T @ avg_pay
        # subtract cost for cooperators
        pay -= self.cost * (self.strategies == 1)
        self.payoffs = pay.astype(np.float32)

    def _update_strategies(self):
        # mask of valid neighbors (exclude self)
        mask = (self.dist_mat <= self.radius) & (self.dist_mat > 0)
        # sample a neighbor for each agent
        # random matrix
        rnd = self.np_random.random((self.N, self.N)) * mask
        chosen = np.argmax(rnd, axis=1)
        # payoffs difference
        delta = self.payoffs[chosen] - self.payoffs
        prob = 1.0 / (1.0 + np.exp(-self.beta * delta))
        draw = self.np_random.random(self.N)
        adopt = draw < prob
        # adopt new strategies
        self.strategies = np.where(adopt, self.strategies[chosen], self.strategies)

    def _get_state(self):
        # individual obs
        obs = np.hstack([
            self.positions / self.world_size,
            self.directions,
            self.strategies[:, None].astype(np.float32),
            self.payoffs[:, None]
        ]).astype(np.float32)
        # shared and graph
        node_obs = obs.copy()
        adj = self.dist_mat.astype(np.float32)
        # rewards
        rewards = self.payoffs.copy()[:, None]
        # done flags & info
        done = self.current_steps >= self.max_steps
        dones = [done] * self.N
        info = {"step_coop_rate": float((self.strategies == 1).sum() / self.N),
                "step_avg_reward": float(self.payoffs.mean()),
                "current_steps": self.current_steps}
        infos = [info] * self.N
        # agent ids
        ids = np.arange(self.N, dtype=np.int32)[:, None]
        # return tuple of lists/arrays matching original shapes
        return (
            list(obs),
            list(ids),
            [node_obs] * self.N,
            [adj] * self.N,
            list(rewards),
            dones,
            infos
        )
