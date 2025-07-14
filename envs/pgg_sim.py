# envs/pgg_sim.py (最终修改版 - 逻辑自洽)

import numpy as np

class PGGSimulator:
    """
    公共品博弈模拟器 (Public Goods Game Simulator) - [k-近邻交互版]

    本版本将博弈和模仿的交互范围，严格限制在智能体半径范围内的k个最近邻居。
    这个k-近邻的定义与 GNN 中使用的图构建逻辑完全一致，从而保证了
    智能体的“认知”（GNN输入）和“现实”（博弈环境）的统一。
    所有修改都封装在本文件内。
    """
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.radius = args.radius
        self.cost = args.cost
        self.r = args.r
        self.beta = args.beta
        self.egt_rounds = args.egt_rounds
        self.egt_steps = args.egt_steps
        
        # [新增] 从args中获取k_neighbors参数
        self.k_neighbors = getattr(args, 'k_neighbors', 0) # 如果没有该参数，默认为0（退化为半径模式）

        self.np_random = np.random.RandomState()

    def seed(self, seed: int):
        self.np_random.seed(seed)

    def _get_k_neighbor_mask(self, adj: np.ndarray) -> np.ndarray:
        """
        [新增] 用NumPy实现与GNN中完全一致的k-近邻图构建逻辑。
        
        返回一个布尔掩码 `interaction_mask` (N, N)，其中 `mask[i, j] = True` 
        表示智能体 j 是 i 在半径内的 k 个最近邻居之一。
        """
        N = self.num_agents
        
        # 如果 k <= 0，则退化为原始的基于半径的交互模式
        if self.k_neighbors <= 0:
            return (adj > 0) & (adj <= self.radius)

        distances = adj.copy()
        # 排除半径外和自身的连接
        mask = (adj == 0) | (adj > self.radius)
        distances[mask] = np.inf

        # 确保 k 不超过邻居总数
        k = min(self.k_neighbors, N - 1)
        
        # 获取每个智能体的前k个最近邻居的索引
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

        # 创建一个掩码，并将对应的k-近邻位置设为True
        interaction_mask = np.zeros_like(adj, dtype=bool)
        rows = np.arange(N).repeat(k)
        cols = k_nearest_indices.flatten()
        
        # 在设置前，需要确保这些邻居不是无穷远（即确实存在半径内的邻居）
        is_valid_neighbor = (distances[rows, cols] != np.inf)
        
        # 只将有效的邻居连接在掩码中设为True
        interaction_mask[rows[is_valid_neighbor], cols[is_valid_neighbor]] = True
        
        return interaction_mask

    def run_simulation(self, initial_strategies: np.ndarray, adj: np.ndarray) -> np.ndarray:
        R, N = self.egt_rounds, self.num_agents
        
        # [核心修改] 在模拟开始前，构建与GNN一致的交互图
        interaction_mask = self._get_k_neighbor_mask(adj)
        
        # 在这个有向的k-近邻图上定义博弈小组和模仿对象
        # 博弈小组：一个智能体i，与所有它能“看到”的邻居j进行博弈
        #           即 interaction_mask[i, j] == True
        #           注意：为了计算小组收益，我们需要考虑小组的完整成员，
        #           这通常是对称的。我们将交互图变为无向图来定义小组。
        group_mask = interaction_mask | interaction_mask.T
        
        # 模仿对象：一个智能体i，只能模仿它能“看到”的邻居j
        #           我们直接使用有向的 interaction_mask 作为 neighbor_mask
        neighbor_mask = interaction_mask

        # 初始化所有轮次的虚拟策略
        virtual_strategies = np.tile(initial_strategies, (R, 1))
        total_cumulative_payoffs = np.zeros((R, N), dtype=np.float32)

        # 主演化循环
        for _ in range(self.egt_steps):
            # 将构建好的掩码传递下去
            step_payoffs = self._compute_payoffs_vectorized(virtual_strategies, group_mask)
            total_cumulative_payoffs += step_payoffs
            virtual_strategies = self._update_strategies_vectorized(virtual_strategies, step_payoffs, neighbor_mask)

        final_average_payoffs = np.mean(total_cumulative_payoffs, axis=0)
        return final_average_payoffs

    def _compute_payoffs_vectorized(self, strategies: np.ndarray, group_mask: np.ndarray) -> np.ndarray:
        """
        向量化计算收益。
        现在接收一个精确定义博弈小组的 group_mask。
        """
        R, N = strategies.shape
        
        # 在小组定义中加入自己
        group_mask_with_self = group_mask | np.eye(N, dtype=bool)

        num_group_members = group_mask_with_self.sum(axis=1)
        num_group_cooperators = strategies @ group_mask_with_self.T
        
        avg_group_payoffs = np.divide(
            num_group_cooperators * self.cost * self.r,
            num_group_members,
            out=np.zeros_like(num_group_cooperators, dtype=np.float32),
            where=(num_group_members != 0)
        )
        
        total_gross_gains = avg_group_payoffs @ group_mask_with_self
        n_games_played = group_mask_with_self.sum(axis=0)
        total_costs = strategies * n_games_played * self.cost
        total_net_gains = total_gross_gains - total_costs

        average_net_payoffs = np.divide(
            total_net_gains,
            n_games_played,
            out=np.zeros_like(total_net_gains),
            where=(n_games_played != 0)
        )
        return average_net_payoffs

    def _update_strategies_vectorized(self, strategies: np.ndarray, payoffs: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
        """
        向量化更新策略。
        现在接收一个精确定义模仿对象的 neighbor_mask。
        """
        R, N = strategies.shape
        
        random_values = self.np_random.random((N, N))
        masked_random = np.where(neighbor_mask, random_values, -np.inf)
        
        # 检查是否有邻居，避免argmax在全为-inf的行上给出无意义的结果
        has_neighbors_mask = np.any(neighbor_mask, axis=1)
        
        imitation_target_indices = np.zeros(N, dtype=int)
        # 只对有邻居的智能体计算argmax
        if np.any(has_neighbors_mask):
            imitation_target_indices[has_neighbors_mask] = np.argmax(masked_random[has_neighbors_mask], axis=1)

        target_strategies = strategies[:, imitation_target_indices]
        target_payoffs = payoffs[:, imitation_target_indices]

        delta_payoffs = target_payoffs - payoffs
        
        # 如果没有邻居，delta_payoff为0，prob_adopt为0.5，这里需要修正
        # 我们让没有邻居的智能体不改变策略
        prob_adopt = np.zeros_like(delta_payoffs)
        if np.any(has_neighbors_mask):
             prob_adopt[:, has_neighbors_mask] = 1 / (1 + np.exp(np.clip(-self.beta * delta_payoffs[:, has_neighbors_mask], -700, 700)))

        should_adopt = self.np_random.random((R, N)) < prob_adopt
        next_strategies = np.where(should_adopt, target_strategies, strategies)
        
        return next_strategies