import cupy as cp
import numpy as np
import random
import math

class MigratoryPGGEnv:
    """
    利用GPU加速的公共物品博弈环境。将智能体状态存储为CuPy数组，
    并采用向量化方式计算所有智能体之间的距离、组收益等，
    从而实现高并发的并行计算。
    """
    metadata = {"render_modes": ["human"], "name": "migratory_pgg_v4_gpu"}

    def __init__(self, N=20, max_cycles=500, size=100, speed=1.0, radius=10.0,
                 cost=1.0, r=1.5, beta=0.5, render_mode=None, visualize=False, seed=None):
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
        self._seed(seed)
        self.reset()

    def _seed(self, seed=None):
        # 设置随机种子，确保CPU、GPU随机性一致
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        np.random.seed(self.seed)
        random.seed(self.seed)
        cp.random.seed(self.seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self.timestep = 0
        # 在GPU上初始化位置（二维均匀分布）
        self.pos = cp.random.uniform(0, self.size, (self.N, 2)).astype(cp.float32)
        # 随机初始化速度方向，保持固定速度大小
        theta = cp.random.uniform(0, 2 * cp.pi, (self.N,))
        self.vel = cp.stack([self.speed * cp.cos(theta), self.speed * cp.sin(theta)], axis=1).astype(cp.float32)
        # 初始化策略：随机选择0或1
        self.strategy = cp.random.randint(0, 2, (self.N,)).astype(cp.int32)
        # 初始化上一次收益为0
        self.last_payoff = cp.zeros((self.N,), dtype=cp.float32)
        return self._compute_observations()

    def step(self, actions):
        """
        输入:
          actions: 字典，键为 "agent_i"，值为对应的动作（二维向量，numpy数组）
        过程:
          - 将动作转换为GPU数组，并归一化
          - 更新速度和位置（利用周期性边界条件）
          - 向量化计算所有智能体之间的距离（考虑周期性边界）
          - 利用邻居矩阵一次性计算每个小组的收益分配
          - 累加所有小组贡献后更新收益
          - 对每个智能体随机选择一个邻居比较收益，并按sigmoid概率更新策略
        返回:
          观测、所有智能体收益、终止标志、截断标志和附加信息
        """
        self.timestep += 1
        # 将actions字典转换为GPU数组（顺序对应agent_0, agent_1, ...）
        actions_array = cp.array([actions[f"agent_{i}"] for i in range(self.N)], dtype=cp.float32)
        # 归一化动作向量，防止除0
        norms = cp.linalg.norm(actions_array, axis=1, keepdims=True) + 1e-8
        directions = actions_array / norms
        # 更新速度和位置（利用周期性边界）
        self.vel = directions * self.speed
        self.pos = (self.pos + self.vel) % self.size

        # 向量化计算所有智能体间的距离
        # 利用广播计算差值矩阵：shape (N, N, 2)
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        # 调整差值以满足周期性边界（映射到 [-size/2, size/2]）
        diff = (diff + self.size / 2) % self.size - self.size / 2
        dists = cp.linalg.norm(diff, axis=2)  # (N, N)

        # 构造邻居矩阵：若距离小于等于 radius，则视为邻居（包括自身）
        M = (dists <= self.radius).astype(cp.float32)

        # 利用向量化计算各个小组（以每个智能体为中心）的组规模和贡献
        group_size = M.sum(axis=1)  # 每个组的智能体数量，shape (N,)
        # 每个小组中合作智能体的贡献（策略为1的贡献cost）
        group_contrib = cp.sum(M * self.strategy[None, :] * self.cost, axis=1)
        # 公共池收益 = 贡献总和 * r
        pool = group_contrib * self.r
        # 每个组中，每个成员分得的收益
        share = pool / group_size

        # 每个智能体参与了多少个小组
        groups_count = M.sum(axis=0)
        # 总收益：每个智能体从所有参与小组获得收益，且每次参与合作都需扣除cost
        all_payoffs = cp.dot(M.T, share) - self.cost * self.strategy * groups_count
        self.last_payoff = all_payoffs

        # 策略更新：对每个智能体随机选择一个邻居进行比较
        # 计算邻居关系（排除自身）
        neighbor_mask = (dists <= self.radius)
        cp.fill_diagonal(neighbor_mask, False)
        # 为方便随机采样，将策略和收益转回CPU
        strategy_cpu = cp.asnumpy(self.strategy)
        payoffs_cpu = cp.asnumpy(all_payoffs)
        neighbor_mask_cpu = cp.asnumpy(neighbor_mask)
        for i in range(self.N):
            candidates = np.where(neighbor_mask_cpu[i])[0]
            if candidates.size == 0:
                continue
            chosen = np.random.choice(candidates)
            delta = payoffs_cpu[chosen] - payoffs_cpu[i]
            prob = 1 / (1 + math.exp(-self.beta * delta))
            if np.random.rand() < prob:
                strategy_cpu[i] = strategy_cpu[chosen]
        # 更新策略回GPU
        self.strategy = cp.array(strategy_cpu, dtype=cp.int32)

        obs = self._compute_observations()
        truncated = {f"agent_{i}": self.timestep >= self.max_cycles for i in range(self.N)}
        terminations = truncated.copy()
        infos = {f"agent_{i}": {} for i in range(self.N)}

        # 将收益转换为CPU数组返回
        return obs, cp.asnumpy(all_payoffs), terminations, truncated, infos

    def _compute_observations(self):
        """
        将当前所有智能体状态转换为观测信息字典
        注：这里仅返回了自身状态，为了节省数据传输，邻居信息可根据需要添加
        """
        obs = {}
        pos_cpu = cp.asnumpy(self.pos)
        vel_cpu = cp.asnumpy(self.vel)
        strategy_cpu = cp.asnumpy(self.strategy)
        payoff_cpu = cp.asnumpy(self.last_payoff)
        for i in range(self.N):
            obs[f"agent_{i}"] = {
                "self": {
                    "position": pos_cpu[i],
                    "velocity": vel_cpu[i],
                    "strategy": int(strategy_cpu[i]),
                    "last_payoff": float(payoff_cpu[i])
                },
                "neighbors": None
            }
        return obs

    def cooperation_rate(self):
        return cp.mean(self.strategy).get()

    def average_payoff(self):
        return cp.mean(self.last_payoff).get()

    def total_payoff(self):
        return cp.sum(self.last_payoff).get()

    def average_cooperator_payoff(self):
        mask = (self.strategy == 1)
        if cp.sum(mask) == 0:
            return 0.0
        return cp.mean(self.last_payoff[mask]).get()

    def average_defector_payoff(self):
        mask = (self.strategy == 0)
        if cp.sum(mask) == 0:
            return 0.0
        return cp.mean(self.last_payoff[mask]).get()

    def cooperation_clustering(self):
        # 计算合作智能体之间的聚集程度
        coop_indices = cp.where(self.strategy == 1)[0]
        coop_positions = self.pos[coop_indices]
        n = coop_positions.shape[0]
        if n <= 1:
            return 0.0
        diff = coop_positions[:, None, :] - coop_positions[None, :, :]
        diff = (diff + self.size / 2) % self.size - self.size / 2
        dists = cp.linalg.norm(diff, axis=2)
        triu_indices = cp.triu_indices(n, k=1)
        mean_dist = cp.mean(dists[triu_indices])
        return (1 / (mean_dist + 1e-6)).get()
