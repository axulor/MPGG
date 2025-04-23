import numpy as np
import random

class RuleBasedPolicy:
    def __init__(self, strategy_name: str = "vicsek", radius: float = 10.0, speed: float = 1.0):
        self.strategy_name = strategy_name
        self.radius = radius
        self.speed = speed

    def select_action(self, agent_id: str, observations: dict) -> np.ndarray:
        if self.strategy_name == "vicsek":
            return self._vicsek(agent_id, observations)
        elif self.strategy_name == "random":
            return self._random()
        elif self.strategy_name == "max_payoff":
            return self._max_payoff(agent_id, observations)
        elif self.strategy_name == "cooperator":
            return self._cooperator(agent_id, observations)
        elif self.strategy_name == "static":
            return self._static(agent_id, observations)
        elif self.strategy_name == "complex_cluster":
            return self._complex_cluster(agent_id, observations)
        else:
            raise ValueError(f"未定义的规则策略: {self.strategy_name}")

    def _vicsek(self, agent_id: str, observations: dict) -> np.ndarray:
        obs = observations[agent_id]
        self_obs = obs["self"]

        neighbors = obs["neighbors"]
        neighbors_vel = [neighbor["velocity"] for neighbor in neighbors]

        if neighbors_vel:
            avg_vel = np.mean(neighbors_vel, axis=0)
            norm = np.linalg.norm(avg_vel)
            if norm > 0:
                return (avg_vel / norm) * self.speed

        current_vel = self_obs["velocity"]
        norm = np.linalg.norm(current_vel)
        if norm > 0:
            return (current_vel / norm) * self.speed

        return self._random()

    def _random(self) -> np.ndarray:
        theta = random.uniform(0, 2 * np.pi)
        return np.array([np.cos(theta), np.sin(theta)]) * self.speed

    def _max_payoff(self, agent_id: str, observations: dict) -> np.ndarray:
        obs = observations[agent_id]
        self_obs = obs["self"]
        neighbors = obs["neighbors"]

        if neighbors:
            best_neighbor = max(neighbors, key=lambda n: n["last_payoff"])
            best_vel = best_neighbor["velocity"]
            norm = np.linalg.norm(best_vel)
            if norm > 0:
                return (best_vel / norm) * self.speed

        current_vel = self_obs["velocity"]
        norm = np.linalg.norm(current_vel)
        if norm > 0:
            return (current_vel / norm) * self.speed

        return self._random()

    def _cooperator(self, agent_id: str, observations: dict) -> np.ndarray:
        obs = observations[agent_id]
        self_obs = obs["self"]
        neighbors = obs["neighbors"]

        cooperator_vels = [n["velocity"] for n in neighbors if n["strategy"] == 1]

        if cooperator_vels:
            avg_vel = np.mean(cooperator_vels, axis=0)
            norm = np.linalg.norm(avg_vel)
            if norm > 0:
                return (avg_vel / norm) * self.speed

        current_vel = self_obs["velocity"]
        norm = np.linalg.norm(current_vel)
        if norm > 0:
            return (current_vel / norm) * self.speed

        return self._random()

    def _static(self, agent_id: str, observations: dict) -> np.ndarray:
        return np.array([0.0, 0.0])

    def _complex_cluster(self, agent_id: str, observations: dict) -> np.ndarray:
        obs = observations[agent_id]
        self_obs = obs["self"]
        neighbors = obs["neighbors"]

        if not neighbors:
            return self._random()

        self_pos = self_obs["position"]
        self_strategy = self_obs["strategy"]

        coop_pos = [n["position"] for n in neighbors if n["strategy"] == 1]
        def_pos = [n["position"] for n in neighbors if n["strategy"] == 0]

        move = np.zeros(2)

        if self_strategy == 1:  # 合作者
            if len(coop_pos) >= 2:
                center_c = np.mean(coop_pos, axis=0)
                move += center_c - self_pos
            else:
                # 周围没有足够合作者，保持位置或缓动
                return self._static(agent_id, observations)

            if def_pos:
                center_d = np.mean(def_pos, axis=0)
                move -= 0.3 * (center_d - self_pos)  # 减小被背叛者吸引

        else:  # 背叛者
            if len(coop_pos) >= 2:
                center_c = np.mean(coop_pos, axis=0)
                move = center_c - self_pos
            else:
                # 背叛者之间保持距离，避免扎堆
                if def_pos:
                    center_d = np.mean(def_pos, axis=0)
                    move = self_pos - center_d  # 远离其他背叛者
                else:
                    return self._random()

        norm = np.linalg.norm(move)
        if norm < 1e-8:
            return self._random()

        return (move / norm) * self.speed

