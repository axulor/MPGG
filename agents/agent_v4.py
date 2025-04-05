import numpy as np

class Agent:
    def __init__(self, agent_id: str, speed: float, cost: float, rng: np.random.Generator):
        self.id = agent_id               # 智能体 ID
        self.speed = speed              # 移动速度
        self.cost = cost                # 公共物品博弈中的贡献成本
        self.rng = rng                  # 随机数生成器（np.random.Generator 类型）

        self.pos = np.zeros(2)          # 当前位置 (x, y)
        self.vel = np.zeros(2)          # 当前速度向量
        self.strategy = 0               # 当前策略（0=背叛，1=合作）
        self.last_payoff = 0.0          # 上一轮收益

    def reset(self, size: float):
        """初始化智能体位置、方向和策略"""
        self.pos = self.rng.random(2) * size
        theta = self.rng.random() * 2 * np.pi
        self.vel = self.speed * np.array([np.cos(theta), np.sin(theta)])
        self.strategy = self.rng.integers(0, 2)
        self.last_payoff = 0.0

    def move(self, direction: np.ndarray):
        """根据给定的方向向量更新速度"""
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.vel = (direction / norm) * self.speed

    def update_position(self, size: float):
        """根据当前速度更新位置，考虑周期边界"""
        self.pos = (self.pos + self.vel) % size

    def observe(self):
        """返回自身状态信息"""
        return {
            "position": self.pos.astype(np.float32),
            "velocity": self.vel.astype(np.float32),
            "strategy": int(self.strategy),
            "last_payoff": float(self.last_payoff),
        }

    def update_strategy(self, other_payoff: float, other_strategy: int, beta: float):
        """使用 Fermi 规则根据另一个个体的策略和收益更新自身策略"""
        delta = other_payoff - self.last_payoff
        prob = 1 / (1 + np.exp(-beta * delta))
        if self.rng.random() < prob:
            self.strategy = other_strategy
