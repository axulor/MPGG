import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
import os
from datetime import datetime

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
        self.run_dir = env.run_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = None
        self.clock = pygame.time.Clock()
        # self.cell_size = self.width // (env.grid_size // env.casino_size)  # 计算赌场格子大小
        self.num_casinos = int(env.grid_size / env.casino_size)  # 计算网格数量（行/列数）
        self.agents = env.agents  # 智能体列表
        self.shuffled_casinos = env.casinos  # 赌场列表

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

    def render(self, t=0, mode="human"):
        """使用 Rectangle + Text 绘制通用可扩展网格，确保刻度严格对齐方格中央, 并将图像保存到 pics/时间戳文件夹下。"""

        fig, ax = plt.subplots(figsize=(4, 4))
        # 计算赌场网格大小
        step_size = 1.0 / self.num_casinos  # 计算每个小方格的边长
        # 确保刻度严格在小方格中央
        grid_centers = np.round(np.linspace(step_size, 1.0, self.num_casinos), decimals=3)

        # 计算赌场边界（用于画网格线）
        grid_lines = np.linspace(0, 1, self.num_casinos + 1)

        # 刻度标签的偏移量（让刻度稍微远离轴）
        offset_factor = 0.2 * step_size  # 适应不同网格大小

        # 绘制网格
        for i in range(self.num_casinos):
            for j in range(self.num_casinos):
                x_min = i * step_size
                y_min = j * step_size
                # 使用 Rectangle 画方格
                rect = patches.Rectangle((x_min, y_min), step_size, step_size, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        # 绘制智能体
        for agent in self.agents:
            c, alpha = self.agents[agent].current_casino  # 赌场坐标
            # 计算赌场所在的网格索引
            row = int((c - step_size) / step_size)  # 修正 row 计算
            col = int((alpha - step_size) / step_size)  # 修正 col 计算
            # 计算该赌场的左下角坐标
            x_min = col * step_size
            y_min = row * step_size
            # 在赌场方格内生成智能体的随机位置
            jitter_x = np.random.uniform(x_min + 0.1 * step_size, x_min + 0.9 * step_size)
            jitter_y = np.random.uniform(y_min + 0.1 * step_size, y_min + 0.9 * step_size)
            # 按策略着色：合作者蓝色，背叛者红色
            color = 'blue' if self.agents[agent].is_cooperator else 'red'
            ax.scatter(jitter_x, jitter_y, c=color, s=40, edgecolors='white')

        # 绘制刻度
        for x in grid_centers:
            # 底部 X 轴刻度
            ax.text(x - step_size / 2, -offset_factor, f"{x:.3f}", ha='center', va='center', fontsize=10)
            # 左侧 Y 轴刻度
            ax.text(-offset_factor, x - step_size / 2, f"{x:.3f}", ha='center', va='center', fontsize=10)

        # 修正刻度线对齐
        ax.set_xticks(grid_centers - step_size / 2)
        ax.set_xticklabels([])
        ax.set_yticks(grid_centers - step_size / 2)
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


        # 保存文件
        file_path = os.path.join(self.run_dir, f"{t}.png")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close(fig)
        # 若需要调试:
        # print(f"Saved figure to {file_path}")
    



    # def render_shuffle(self, t=0, mode="human"):
    #     """
    #     在保留原有网格外观的前提下，将 (c, alpha) 随机打乱分配到
    #     num_casinos x num_casinos 的方格里，并在方格内画出对应赌场与智能体。
    #     """
    #     fig, ax = plt.subplots(figsize=(4, 4))

    #     # === 1) 网格基本参数 ===
    #     step_size = 1.0 / self.num_casinos
    #     grid_lines = np.linspace(0, 1, self.num_casinos + 1)

    #     # 绘制网格方格（和之前一样）
    #     for i in range(self.num_casinos):
    #         for j in range(self.num_casinos):
    #             x_min = j * step_size
    #             y_min = i * step_size
    #             rect = patches.Rectangle((x_min, y_min),
    #                                     step_size, step_size,
    #                                     linewidth=1, edgecolor='black', facecolor='none')
    #             ax.add_patch(rect)

    #     # === 2) 在每个方格内放置随机分配到的 (c, alpha) 并写文本 ===
    #     # 这里假设 self.shuffled_casinos = [(c1, alpha1), (c2, alpha2), ...]
    #     # 长度 = num_casinos * num_casinos
    #     # 格子索引 row=i, col=j => index = i * num_casinos + j
    #     for i in range(self.num_casinos):
    #         for j in range(self.num_casinos):
    #             index = i * self.num_casinos + j
    #             c, alpha = self.shuffled_casinos[index]

    #             x_min = j * step_size
    #             y_min = i * step_size

    #             # 把 (c, alpha) 的数值显示到格子中央
    #             center_x = x_min + 0.5 * step_size
    #             center_y = y_min + 0.5 * step_size
    #             ax.text(center_x, center_y,
    #                     f"{c:.2f}\n{alpha:.2f}",
    #                     ha='center', va='center',
    #                     fontsize=8, color='black')

    #     # === 3) 绘制智能体: 找到每个智能体所在的 (c, alpha)，再查字典获取该方格位置 ===
    #     for agent_name, agent_obj in self.agents.items():
    #         c, alpha = agent_obj.current_casino

    #         # 用一个字典 casino_to_rc 来找 row, col
    #         # 如果 c, alpha 是浮点，可能需要处理精度
    #         (row, col) = (c, alpha)  # 或者 (round(c,3),round(alpha,3))

    #         # 计算该方格左下角
    #         x_min = col * step_size
    #         y_min = row * step_size

    #         # 在方格内随机抖动
    #         jitter_x = np.random.uniform(x_min + 0.1 * step_size, x_min + 0.9 * step_size)
    #         jitter_y = np.random.uniform(y_min + 0.1 * step_size, y_min + 0.9 * step_size)

    #         # 根据合作/背叛着色
    #         color = 'blue' if agent_obj.is_cooperator else 'red'
    #         ax.scatter(jitter_x, jitter_y, c=color, s=40, edgecolors='white')

    #     # === 4) 设置外观，如轴范围、网格线等 ===
    #     # 与原先类似
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.set_xticks(grid_lines, minor=True)
    #     ax.set_yticks(grid_lines, minor=True)
    #     ax.grid(True, which='minor', linestyle="--", alpha=0.5)

    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_position(('data', 0))
    #     ax.spines['left'].set_position(('data', 0))

    #     legend_elements = [
    #         Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Cooperator'),
    #         Line2D([0], [0], marker='o', color='w', markerfacecolor='red',  markersize=8, label='Defector')
    #     ]
    #     ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=12)
    #     ax.set_title("Shuffled Grid Layout")

    #     # === 5) 保存到文件而不显示 ===
    #     file_path = os.path.join(self.run_dir, f"{t}.png")
    #     plt.tight_layout()
    #     plt.savefig(file_path)
    #     plt.close(fig)
    #     # print(f"Saved figure: {file_path}")