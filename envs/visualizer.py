import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
import os
from datetime import datetime

class PygameVisualizer:
    """Pygame 网格环境可视化，支持智能体动态移动
       环境中每个网格的参数由 (c, r) 给出，其中 c 为固定成本，r 为增益倍率
    """

    def __init__(self, env, width=600, height=600, fps=60):
        """
        初始化可视化器
        :param env: 修改后的 MigratoryPGGEnv 环境对象
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
        # 计算网格数量（每边格子数），使用 env.cell_size
        self.num_cells = int(env.grid_size / env.cell_size)
        self.agents = env.agents  # 智能体列表
        # 将环境中手动定义的网格参数存储在字典中（键为 (row, col)，值为 (c, r)）
        self.grid_keys = list(env.grids.keys())

    def _initialize_pygame(self):
        """初始化 Pygame 窗口"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Migratory Public Goods Game")

    def render(self, t=0, mode="human"):
        """
        使用 Matplotlib 绘制网格及智能体分布：
          - 每个网格由 (row, col) 标识，并绑定参数 (c, r)
          - 绘制时利用 grid_id 确定坐标，并在每个格子内加入随机偏移（jitter）以展示智能体位置
          - 横轴表示增益倍率 r，纵轴表示固定成本 c
        最终图像保存至指定文件夹中。
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        # 每个网格的边长（归一化至 1）
        step_size = 1.0 / self.num_cells  
        # 计算网格中心位置（用于刻度显示）
        grid_centers = np.round(np.linspace(step_size / 2, 1.0 - step_size / 2, self.num_cells), decimals=3)
        # 网格边界
        grid_lines = np.linspace(0, 1, self.num_cells + 1)
        offset_factor = 0.2 * step_size

        # 绘制网格背景
        for i in range(self.num_cells):
            for j in range(self.num_cells):
                x_min = j * step_size
                y_min = i * step_size
                rect = patches.Rectangle((x_min, y_min), step_size, step_size, linewidth=1,
                                         edgecolor='black', facecolor='none')
                ax.add_patch(rect)

        # 绘制智能体
        for agent in self.agents:
            # 当前网格编号 (row, col)
            grid_id = self.agents[agent].current_grid
            # 从环境中获取该网格的参数 (c, r)
            c, r = self.env.grids[grid_id]
            row, col = grid_id
            # 计算格子左下角坐标
            x_min = col * step_size
            y_min = row * step_size
            # 在格子内生成随机位置（加入 jitter）
            jitter_x = np.random.uniform(x_min + 0.1 * step_size, x_min + 0.9 * step_size)
            jitter_y = np.random.uniform(y_min + 0.1 * step_size, y_min + 0.9 * step_size)
            # 根据智能体策略选择颜色：蓝色为合作，红色为背叛
            color = 'blue' if self.agents[agent].is_cooperator else 'red'
            ax.scatter(jitter_x, jitter_y, c=color, s=40, edgecolors='white')

        # 绘制刻度标签：用网格中心坐标作为刻度
        for x in grid_centers:
            ax.text(x, -offset_factor, f"{x:.3f}", ha='center', va='center', fontsize=10)
            ax.text(-offset_factor, x, f"{x:.3f}", ha='center', va='center', fontsize=10)

        ax.set_xticks(grid_lines)
        ax.set_yticks(grid_lines)
        ax.grid(True, which='both', linestyle="--", alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Cooperator'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Defector')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=12)
        # 注释：横轴代表增益倍率 r，纵轴代表固定成本 c
        ax.annotate(r"$r$", xy=(0.90, -0.10), xycoords='axes fraction', fontsize=20, ha='left', va='center')
        ax.annotate(r"$c$", xy=(-0.10, 0.90), xycoords='axes fraction', fontsize=20, ha='center', va='bottom')

        # 保存图像
        file_path = os.path.join(self.run_dir, f"{t}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close(fig)
