# envs/pgg_sim.py (高性能向量化版 - 收益计算修正)

import numpy as np

class PGGSimulator:
    """
    公共品博弈模拟器 (Public Goods Game Simulator) - [高性能向量化版]

    本版本通过 NumPy 向量化操作，并行处理所有 egt_rounds 的模拟，
    旨在大幅提升单个 env.step() 的计算速度。
    核心思想是将 for 循环替换为矩阵运算。
    [v2] 修正了合作者成本的计算逻辑，使其与参与的博弈次数精确挂钩。
    """
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r
        self.beta = args.beta
        self.egt_rounds = args.egt_rounds
        self.egt_steps = args.egt_steps
        self.np_random = np.random.RandomState()

    def seed(self, seed: int):
        self.np_random.seed(seed)

    def run_simulation(self, initial_strategies: np.ndarray, adj: np.ndarray) -> np.ndarray:
        R, N = self.egt_rounds, self.num_agents
        
        # 预计算邻居关系
        neighbor_mask = (adj > 0) & (adj <= self.radius)
        group_mask = (adj <= self.radius)
        
        # 初始化所有轮次的虚拟策略
        virtual_strategies = np.tile(initial_strategies, (R, 1))
        
        total_cumulative_payoffs = np.zeros((R, N), dtype=np.float32)

        # 主演化循环
        for _ in range(self.egt_steps):
            step_payoffs = self._compute_payoffs_vectorized(virtual_strategies, group_mask)
            total_cumulative_payoffs += step_payoffs
            virtual_strategies = self._update_strategies_vectorized(virtual_strategies, step_payoffs, neighbor_mask)

        final_average_payoffs = np.mean(total_cumulative_payoffs, axis=0)
        return final_average_payoffs

    def _compute_payoffs_vectorized(self, strategies: np.ndarray, group_mask: np.ndarray) -> np.ndarray:
        """
        [修正版] 向量化计算所有轮次的单步平均收益。
        合作者的成本现在是根据其参与的博弈次数精确计算的。

        Args:
            strategies (np.ndarray): 形状为 (R, N) 的策略矩阵。
            group_mask (np.ndarray): 形状为 (N, N) 的博弈组关系掩码。

        Returns:
            np.ndarray: 形状为 (R, N) 的收益矩阵。
        """
        R, N = strategies.shape
        
        # --- 1. 计算毛收益 (Gross Gain) ---
        
        # a. 计算每个博弈组的成员数量, 形状 (N,)
        num_group_members = group_mask.sum(axis=1)

        # b. 计算每个博弈组中的合作者数量, 形状 (R, N)
        num_group_cooperators = strategies @ group_mask.T
        
        # c. 计算每个小组产生的平均收益, 形状 (R, N)
        avg_group_payoffs = np.divide(
            num_group_cooperators * self.cost * self.r,
            num_group_members,
            out=np.zeros_like(num_group_cooperators, dtype=np.float32),
            where=(num_group_members != 0)
        )
        
        # d. 计算每个智能体从其参与的所有小组中获得的“总毛收益”
        #    这是该智能体所有收益的来源
        total_gross_gains = avg_group_payoffs @ group_mask # Shape (R, N)

        # --- 2. 计算总成本 (Total Cost) ---

        # a. 计算每个智能体参与的博弈次数，这是一个与策略无关的静态量, 形状 (N,)
        n_games_played = group_mask.sum(axis=0)

        # b. 计算每个合作者需要付出的总成本
        #    `strategies` (R, N) 矩阵中为1的地方是合作者
        #    `n_games_played` (N,) 向量是每个位置的参与次数
        #    通过广播，我们可以得到每个合作者在所有轮次中的总成本
        total_costs = strategies * n_games_played * self.cost # Shape (R, N)
        # 这个操作意味着：对于 (r, i) 位置，如果 strategies[r, i] 是 0 (背叛者)，成本为0。
        # 如果是 1 (合作者)，成本为 n_games_played[i] * c。

        # --- 3. 计算净收益 (Net Gain) ---
        total_net_gains = total_gross_gains - total_costs # Shape (R, N)

        # --- 4. 计算平均净收益 ---
        average_net_payoffs = np.divide(
            total_net_gains,
            n_games_played,
            out=np.zeros_like(total_net_gains),
            where=(n_games_played != 0)
        )
        
        return average_net_payoffs

    def _update_strategies_vectorized(self, strategies: np.ndarray, payoffs: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
        """
        向量化更新所有轮次的策略。(此函数逻辑不变)
        """
        R, N = strategies.shape
        
        # Gumbel-Max 技巧选择模仿对象
        random_values = self.np_random.random((N, N))
        masked_random = np.where(neighbor_mask, random_values, -np.inf)
        imitation_target_indices = np.argmax(masked_random, axis=1)
        has_neighbors = neighbor_mask.sum(axis=1) > 0
        imitation_target_indices[~has_neighbors] = np.arange(N)[~has_neighbors]

        # 获取模仿对象的策略和收益
        target_strategies = strategies[:, imitation_target_indices]
        target_payoffs = payoffs[:, imitation_target_indices]

        # 计算采纳概率并决定是否更新
        delta_payoffs = target_payoffs - payoffs
        prob_adopt = 1 / (1 + np.exp(np.clip(-self.beta * delta_payoffs, -700, 700)))
        should_adopt = self.np_random.random((R, N)) < prob_adopt
        
        # 生成新策略
        next_strategies = np.where(should_adopt, target_strategies, strategies)
        
        return next_strategies