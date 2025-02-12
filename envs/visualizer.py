import pygame
import numpy as np

class PygameVisualizer:
    """Pygame 赌场环境可视化，支持智能体动态移动"""

    def __init__(self, env, width=600, height=600, fps=60):
        """
        初始化可视化器
        :param env: 迁移公共物品博弈环境 (MigratoryPGGEnv)
        :param width: 窗口宽度
        :param height: 窗口高度
        :param fps: 最大帧率
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = None
        self.clock = pygame.time.Clock()
        self.cell_size = self.width // (env.grid_size // env.casino_size)  # 计算赌场格子大小

    def _initialize_pygame(self):
        """初始化 Pygame 窗口"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Migratory Public Goods Game")

    def set_current_casino(self, casino):
        """更新赌场位置，同时保存上一次的位置"""
        if self.current_casino is not None:
            self.last_position = self.current_casino  # 记录上一帧位置
        else:
            self.last_position = (0.0, 0.0)  # 避免 None，给默认初始值
        self.current_casino = casino

    def _draw_grid(self):
        """绘制赌场网格背景"""
        self.screen.fill((255, 255, 255))  # 清空屏幕（白色背景）
        num_cells = self.env.grid_size // self.env.casino_size  # 赌场网格数量

        for i in range(num_cells):
            for j in range(num_cells):
                pygame.draw.rect(
                    self.screen, (200, 200, 200),  # 灰色网格
                    (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size),
                    1  # 线宽
                )

    def render(self):
        """让智能体从上一个赌场平滑移动到新赌场"""
        if self.screen is None:
            self._initialize_pygame()

        self._draw_grid()  

        num_frames = 10  # 让动画有 10 帧

        for frame in range(num_frames):
            self.screen.fill((255, 255, 255))
            self._draw_grid()

            for agent in self.env.agent_objects:
                old_x, old_y = agent.last_position
                new_x, new_y = agent.current_casino

                # 计算中间位置（线性插值）
                px = int((old_x * (num_frames - frame) + new_x * frame) / num_frames * self.width)
                py = int((old_y * (num_frames - frame) + new_y * frame) / num_frames * self.height)

                color = (0, 0, 255) if agent.is_cooperator else (255, 0, 0)
                pygame.draw.circle(self.screen, color, (px, py), 10)

            pygame.display.flip()
            self.clock.tick(60)  # 保持 60 FPS

    def render(self, mode="human"):
        """使用 Rectangle + Text 绘制通用可扩展网格，确保刻度严格对齐方格中央"""
        fig, ax = plt.subplots(figsize=(4, 4))
        # 计算赌场网格大小
        num_casinos = int(self.grid_size / self.casino_size)  # 计算网格数量（行/列数）
        step = 1.0 / num_casinos  # 计算每个小方格的边长
        # 确保刻度严格在小方格中央
        grid_centers = np.round(np.linspace(step, 1.0, num_casinos), decimals=3)

        # 计算赌场边界（用于画网格线）
        grid_lines = np.linspace(0, 1, num_casinos + 1)

        # 刻度标签的偏移量（让刻度稍微远离轴）
        offset_factor = 0.2 * step  # 适应不同网格大小

        # 绘制网格
        for i in range(num_casinos):
            for j in range(num_casinos):
                x_min = i * step
                y_min = j * step
                # 使用 Rectangle 画方格
                rect = patches.Rectangle((x_min, y_min), step, step, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        # 绘制智能体
        for agent in self.agent_objects:
            c, alpha = agent.current_casino  # 赌场坐标
            # 计算赌场所在的网格索引
            row = int((c - step) / step)  # 修正 row 计算
            col = int((alpha - step) / step)  # 修正 col 计算
            # 计算该赌场的左下角坐标
            x_min = col * step
            y_min = row * step
            # 在赌场方格内生成智能体的随机位置
            jitter_x = np.random.uniform(x_min + 0.1 * step, x_min + 0.9 * step)
            jitter_y = np.random.uniform(y_min + 0.1 * step, y_min + 0.9 * step)
            # 按策略着色：合作者蓝色，背叛者红色
            color = 'blue' if agent.is_cooperator else 'red'
            ax.scatter(jitter_x, jitter_y, c=color, s=40, edgecolors='white')

        # 绘制刻度
        for x in grid_centers:
            # 底部 X 轴刻度
            ax.text(x - step / 2, -offset_factor, f"{x:.3f}", ha='center', va='center', fontsize=10)
            # 左侧 Y 轴刻度
            ax.text(-offset_factor, x - step / 2, f"{x:.3f}", ha='center', va='center', fontsize=10)

        # 修正刻度线对齐
        ax.set_xticks(grid_centers - step / 2)
        ax.set_xticklabels([])
        ax.set_yticks(grid_centers - step / 2)
        ax.set_yticklabels([])

        # 网格线设置
        ax.set_xticks(grid_lines, minor=True)
        ax.set_yticks(grid_lines, minor=True)
        ax.grid(True, which='minor', linestyle="--", alpha=0.5)

        # 修正坐标轴边框 & 刻度线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', 0))

        # 确保坐标范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Cooperator'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Defector')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=12)
        ax.annotate(r"$c$", xy=(0.90, -0.05), xycoords='axes fraction', fontsize=20, ha='left', va='center')
        ax.annotate(r"$\alpha$", xy=(-0.05, 0.90), xycoords='axes fraction', fontsize=20, ha='center', va='bottom')

        # 显示图像
        if mode == "human":
            plt.tight_layout()
            plt.show()
        else:
            print("please use human mode")