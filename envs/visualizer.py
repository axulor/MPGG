import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from datetime import datetime

class PygameVisualizer:
    """
    可视化器：用于生成环境快照（不显示晶格网络的连边）。
    
    快照内容：
      1. 绘制整个归一化区域内的晶格网络节点（非常小的灰色实心点），使得网格内部看起来布满了格点。
      2. 绘制网格边界，将区域划分为网格；外框线为实线，内部网格线为虚线，
         标签分别放置在整体网格下方中央（显示列号）和左侧中央（显示行号）。
      3. 对于每个智能体，检查其所在节点，将对应节点的灰色点改为实心圆；
         如果同一节点上有多个智能体，则该节点的点尺寸按数量增大（但不超过设定上限）。
      4. 当同一节点上既有合作也有背叛时，根据合作比例 p 线性混合颜色（颜色 = (1-p, 0, p)）。
    """

    def __init__(self, env, width=600, height=600):
        """
        初始化可视化器
        :param env: 环境对象，要求具有以下属性：
                    - env.network_size: 晶格网络的边长（节点数），如80表示80×80的节点矩阵
                    - env.grid_division: 网格划分数（例如8表示8×8的网格）
                    - env.run_dir: 存储快照图片的目录
                    - env.nodes: 字典，键为 (row, col) 的晶格节点坐标
                    - env.agents: 字典，所有智能体对象，每个智能体要求有属性 current_node（晶格节点坐标）和 is_cooperator（策略）
        :param width: 图像宽度（用于显示，可选）
        :param height: 图像高度（用于显示，可选）
        """
        self.env = env
        self.run_dir = env.run_dir  # 快照图片保存目录
        self.width = width
        self.height = height

    def render(self, t=0):
        """
        生成环境快照：
          - 绘制晶格网络的所有节点（非常小的灰色实心点，表示格点）
          - 绘制网格边界，其中外框线为实线，内部网格线为虚线；并在整体横轴下方显示各列编号（0到grid_div-1），在整体左侧显示各行编号（0到grid_div-1）
          - 根据智能体位置将对应节点绘制为实心圆，其尺寸根据重叠数量增大（但不超过设定上限），颜色根据合作比例线性混合（无边框）
        :param t: 当前时间步（用于保存文件名和标题）
        """
        # 确保 self.run_dir 非空
        if self.run_dir is None:
            root_folder = "pics"
            os.makedirs(root_folder, exist_ok=True)
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(root_folder, timestamp_folder)
            os.makedirs(self.run_dir, exist_ok=True)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # ---------------------------
        # 1. 绘制晶格网络的所有节点（非常小的灰色实心点）
        network_size = self.env.network_size  # 节点矩阵大小
        for (row, col) in self.env.nodes.keys():
            # 居中坐标：节点从 (col, row) 转换，加 0.5 使其居中
            x = (col + 0.5) / network_size
            y = (network_size - 1 - row + 0.5) / network_size
            # 使用非常小的 marker 绘制实心点，颜色灰色，无边框
            ax.plot(x, y, marker='o', markersize=1, markerfacecolor='grey', markeredgecolor='none', zorder=1)
        
        # ---------------------------
        # 2. 绘制网格边界与整体编号标签
        grid_div = self.env.grid_division  # 例如8
        cell_size = 1.0 / grid_div         # 每个网格单元的大小
        
        # 绘制垂直边界
        for i in range(grid_div + 1):
            x_line = i * cell_size
            if i == 0 or i == grid_div:
                # 外框实线
                ax.plot([x_line, x_line], [0, 1], color='black', linewidth=1, linestyle='solid', zorder=2)
            else:
                # 内部边界：虚线、浅灰色、线宽细一些
                ax.plot([x_line, x_line], [0, 1], color='black', linewidth=0.5, linestyle='dashed', zorder=2)

        # 绘制水平边界
        for j in range(grid_div + 1):
            y_line = j * cell_size
            if j == 0 or j == grid_div:
                ax.plot([0, 1], [y_line, y_line], color='black', linewidth=1, linestyle='solid', zorder=2)
            else:
                ax.plot([0, 1], [y_line, y_line], color='black', linewidth=0.5, linestyle='dashed', zorder=2)

        
        offset = 0.02  # 标签偏移量
        # 绘制横轴标签：在整体底部每个网格单元中央显示列号（0到grid_div-1）
        for j in range(grid_div):
            x_center = j * cell_size + cell_size / 2
            ax.text(x_center, -offset, f"{j}", ha='center', va='top', fontsize=8, zorder=3)
        # 绘制纵轴标签：在整体左侧每个网格单元中央显示行号（0到grid_div-1）
        for i in range(grid_div):
            y_center = i * cell_size + cell_size / 2
            ax.text(-offset, y_center, f"{i}", ha='right', va='center', fontsize=8, zorder=3)
        
        # ---------------------------
        # 3. 根据智能体位置更新节点显示
        # 统计每个节点上智能体的数量以及策略分布
        node_counts = {}       # 键：节点 (row, col)，值：智能体数量
        node_strategies = {}   # 键：节点，值：策略列表（True 表示合作，False 表示背叛）
        for agent in self.env.agents.values():
            node = agent.current_node  # 假设格式为 (row, col)
            if node not in node_counts:
                node_counts[node] = 0
                node_strategies[node] = []
            node_counts[node] += 1
            node_strategies[node].append(agent.is_cooperator)
        
        # 定义节点显示尺寸参数
        base_marker_size = 25   # 基础尺寸
        size_scale = 15         # 每增加一个智能体增加的尺寸
        max_marker_size = 70   # 最大尺寸
        
        # 对于每个有智能体的节点，用实心圆表示，颜色根据合作比例混合（无边框）
        for node, count in node_counts.items():
            row, col = node
            x = (col + 0.5) / network_size
            y = (network_size - 1 - row + 0.5) / network_size
            marker_size = base_marker_size + size_scale * (count - 1)
            marker_size = min(marker_size, max_marker_size)
            # 计算合作比例 p
            strategies = node_strategies[node]
            p = sum(strategies) / len(strategies)  # p为合作比例
            # 混合颜色：p=1时蓝色 (0,0,1)，p=0时红色 (1,0,0)
            color = (1 - p, 0, p)
            ax.scatter(x, y, s=marker_size, c=[color], edgecolors='none', zorder=4)
        
        # ---------------------------
        # 图形调整与保存
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Environment Snapshot at t = {t}")
        plt.tight_layout()
        
        # 保存图像到指定目录
        file_path = os.path.join(self.run_dir, f"{t}.png")
        plt.savefig(file_path, dpi=300)
        plt.close(fig)
