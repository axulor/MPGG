import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

class PublicGoodsGame:
    def __init__(self, grid_size=10, num_agents=300, r_min=1.1, r_max=5.2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.r_min = r_min
        self.r_max = r_max

        # 生成赌场参数
        x_indices, y_indices = np.meshgrid(np.linspace(0.1, 1.0, grid_size), np.linspace(0.1, 1.0, grid_size))
        self.r_matrix = y_indices  # Y 轴：增益因子（需要归一化计算 r_eff）
        self.c_matrix = x_indices  # X 轴：成本

        # 计算每个赌场初始的智能体数量
        agents_per_casino = num_agents // (grid_size * grid_size)
        extra_agents = num_agents % (grid_size * grid_size)

        all_casinos = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        np.random.shuffle(all_casinos)

        self.initial_population_matrix = np.zeros((grid_size, grid_size), dtype=int)  # 存储初始人口分布

        self.agents = []
        for x, y in all_casinos:
            num_agents_here = agents_per_casino + (1 if extra_agents > 0 else 0)
            extra_agents -= 1
            self.initial_population_matrix[x, y] = num_agents_here  # 存储初始智能体数量

            for _ in range(num_agents_here):
                strategy = np.random.choice(['C', 'D'])
                self.agents.append({'x': x, 'y': y, 'strategy': strategy})


            
    def get_agents_at(self, x, y):
        return [a for a in self.agents if a['x'] == x and a['y'] == y]

    def play_game(self, K=0.1):
        """ 每个智能体基于 Fermi 规则决定是否改变策略，并计算收益 """

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                agents_here = self.get_agents_at(x, y)
                if len(agents_here) == 0:
                    continue  # 跳过无人赌场

                # 获取当前赌场的初始智能体数量 n_m(0)
                initial_population = self.initial_population_matrix[x, y]

                # **计算 PGG 博弈增益因子 r_eff**
                r_alpha = self.r_matrix[x, y]  # 赌场的原始增益参数 α
                r_eff = self.r_min + (self.r_max - self.r_min) * r_alpha  # 计算 r_eff

                # 获取当前赌场的成本参数 c
                c = self.c_matrix[x, y]

                # **统计合作者人数**
                num_cooperators = sum(1 for a in agents_here if a['strategy'] == 'C')
                num_total = len(agents_here)

                # **计算收益**
                for agent in agents_here:
                    if agent['strategy'] == 'C':
                        agent['payoff'] = (r_eff * num_cooperators * c) / num_total - c  # 合作收益
                    else:
                        agent['payoff'] = (r_eff * num_cooperators * c / num_total)  # 叛徒收益

                # **Fermi 规则进行策略更新**
                for agent in agents_here:
                    if len(agents_here) > 1:  # 至少有一个邻居
                        # **随机选择一个邻居**
                        neighbor = np.random.choice(agents_here)

                        # **计算 Fermi 规则的转换概率**
                        P = 1 / (1 + np.exp((agent['payoff'] - neighbor['payoff']) / K))

                        # **根据概率 P 进行策略更新**
                        if np.random.rand() < P:
                            agent['strategy'] = neighbor['strategy']  # 复制邻居策略



    def move_agents(self):
        """ 每个智能体随机选择一个周围赌场移动（或者原地不动） """

        for agent in self.agents:
            x, y = agent['x'], agent['y']

            # **定义5种可能的移动方式**
            possible_moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # 原地不动, 上, 下, 左, 右
            
            while True:
                dx, dy = possible_moves[np.random.randint(len(possible_moves))]  # 随机选择一个方向
                nx, ny = x + dx, y + dy  # 计算新位置

                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    agent['x'], agent['y'] = nx, ny  # 移动到新位置
                    break  # 成功移动后跳出循环

    def plot_agents(self):
        """ 绘制当前所有赌场及智能体分布，确保赌场刻度和智能体位置正确 """

        if len(self.agents) == 0:  
            print("Warning: No agents to plot!")
            return

        # **关闭所有空白 Figure，避免 Matplotlib 生成多个 Figure**
        plt.close('all')

        # **确保使用 Figure 1**
        fig, ax = plt.subplots(figsize=(6, 6), num=1)


        # **修正网格范围，使赌场坐标范围为 `[0,1]`**
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # **设置赌场网格刻度，使其符合 `[0,1]`，刻度位于小方格的正下方和正左方**
        tick_positions = np.linspace(0, 1, self.grid_size + 1)  # 赌场边界刻度
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

        ax.grid(True, linestyle="-", linewidth=1.5, color="black")  # 加强赌场边界

        # **设置刻度标签，使其对齐赌场内部**
        tick_labels = np.round(np.linspace(0.0, 1.0, self.grid_size + 1), 2)  # 生成刻度标签
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_yticklabels(tick_labels, fontsize=10)

        # # **设置标题**
        # ax.set_title("Current Agent Distribution (Scaled 0-1)", pad=30)  # 向上移动标题以避免重叠

        # **存储智能体坐标**
        cooperators_x, cooperators_y = [], []
        defectors_x, defectors_y = [], []

        for agent in self.agents:
            x, y = agent['x'], agent['y']

            # **计算当前赌场的左下角坐标**
            casino_x = self.c_matrix[x, y] - (1 / self.grid_size) / 2  # 赌场中心偏移
            casino_y = self.r_matrix[x, y] - (1 / self.grid_size) / 2

            # **智能体随机分布，但不贴边界**
            jitter_range = (1 / self.grid_size) * 0.7  # 让智能体在赌场范围内移动但不碰边
            jitter_x = np.random.uniform(-jitter_range / 2, jitter_range / 2)
            jitter_y = np.random.uniform(-jitter_range / 2, jitter_range / 2)

            plot_x = casino_x + jitter_x
            plot_y = casino_y + jitter_y

            if agent['strategy'] == 'C':
                cooperators_x.append(plot_x)
                cooperators_y.append(plot_y)
            else:
                defectors_x.append(plot_x)
                defectors_y.append(plot_y)

        # **确保有数据**
        if len(cooperators_x) > 0:
            ax.scatter(cooperators_x, cooperators_y, c='b', s=50, alpha=0.8, edgecolors="black", label="Cooperators (C)")
        if len(defectors_x) > 0:
            ax.scatter(defectors_x, defectors_y, c='r', s=50, alpha=0.8, edgecolors="black", label="Defectors (D)")

        # **添加图例**
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        # **防止窗口闪退**
        plt.show(block=True)  # **确保窗口保持打开**


    
game = PublicGoodsGame(grid_size=10, num_agents=300)

# 运行5个时间步
for step in range(20):
    print(f"Step {step + 1}")
    game.plot_agents()  # 观察分布
    game.play_game()  # 进行博弈
    game.move_agents()  # 智能体移动


