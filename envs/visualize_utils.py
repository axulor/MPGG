# envs/visualize_utils.py

import numpy as np
import matplotlib
# 使用 'Agg' 后端，确保即使在没有图形界面的服务器上也能正常渲染到内存
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from typing import Dict

# --- 全局变量来管理matplotlib对象 ---
# 这样可以复用同一个窗口/画布，大大提高连续渲染多帧的速度
_fig = None
_ax = None

def _draw_periodic_line(ax, p1, p2, world_size):
    """
    绘制考虑周期性边界的线条。
    如果两点间的距离在x或y轴上超过世界的一半，就分段绘制。
    """
    delta = p1 - p2
    # 处理X轴周期性
    if np.abs(delta[0]) > world_size / 2:
        if p1[0] < p2[0]:
            p1_prime = p1 + np.array([world_size, 0])
            p2_prime = p2 - np.array([world_size, 0])
        else:
            p1_prime = p1 - np.array([world_size, 0])
            p2_prime = p2 + np.array([world_size, 0])
        ax.plot([p1[0], p2_prime[0]], [p1[1], p2_prime[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
        ax.plot([p1_prime[0], p2[0]], [p1_prime[1], p2[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
    # 处理Y轴周期性 (可以与X轴同时发生)
    elif np.abs(delta[1]) > world_size / 2:
        if p1[1] < p2[1]:
            p1_prime = p1 + np.array([0, world_size])
            p2_prime = p2 - np.array([0, world_size])
        else:
            p1_prime = p1 - np.array([0, world_size])
            p2_prime = p2 + np.array([0, world_size])
        ax.plot([p1[0], p2_prime[0]], [p1[1], p2_prime[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
        ax.plot([p1_prime[0], p2[0]], [p1_prime[1], p2[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)
    # 正常绘制
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)


def render_frame(
    render_data: Dict[str, np.ndarray], 
    world_size: float,
    radius: float, # [NEW] 接收 radius 参数 
    current_steps: int
) -> np.ndarray:
    """
    根据传入的数据字典，渲染一帧环境状态的图像。
    这个函数包含了绘制智能体、交互连线和状态文本的所有逻辑。

    Args:
        render_data (dict): 包含 'positions', 'strategies', 'payoffs', 'adj' 的字典。
        world_size (float): 环境世界的边界大小。
        current_steps (int): 当前的步数，用于显示在标题中。

    Returns:
        np.ndarray: 一个 (H, W, 3) 的 RGB 图像数组。
    """
    global _fig, _ax
    
    # 第一次调用时，创建matplotlib的figure和axes对象
    if _fig is None or _ax is None:
        _fig, _ax = plt.subplots(figsize=(10, 10), dpi=120) # 提高dpi以获得更清晰的图像

    # 清除上一帧的绘图内容，为绘制新一帧做准备
    _ax.clear()

    # --- 1. 从数据字典中安全地解包数据 ---
    positions = render_data.get('positions')
    strategies = render_data.get('strategies')
    adj = render_data.get('adj')

    # 安全检查：如果核心数据缺失，则无法绘图
    if positions is None or strategies is None or adj is None:
        print("Warning: render_frame received incomplete data (positions, strategies, or adj is missing). Cannot render.")
        w, h = _fig.canvas.get_width_height()
        return np.zeros((h, w, 3), dtype=np.uint8)

    # --- 2. 绘制交互连线 (边) ---
    # 找到邻接矩阵中所有表示连接的非零元素
    src_nodes, dst_nodes = np.where((adj > 0) & (adj <= radius))
    # 遍历所有边，并绘制线条
    for src, dst in zip(src_nodes, dst_nodes):
        # 为了避免重复绘制 (e.g., 0->1 和 1->0)，只绘制索引小到大的边
        if src < dst:
            _draw_periodic_line(_ax, positions[src], positions[dst], world_size)

    # --- 3. 绘制智能体 (节点) ---
    # 根据策略（0=背叛者, 1=合作者）分离数据
    coop_mask = (strategies == 1)
    defect_mask = (strategies == 0)

    coop_pos = positions[coop_mask]
    defect_pos = positions[defect_mask]
    
    # 绘制合作者 (蓝色圆形)
    if len(coop_pos) > 0:
        _ax.scatter(coop_pos[:, 0], coop_pos[:, 1], 
                     s=150, c='#2196F3', alpha=0.9, edgecolors='w', linewidths=1.0, 
                     label=f'Cooperators ({len(coop_pos)})', zorder=2)
    
    # 绘制背叛者 (红色'X'形)
    if len(defect_pos) > 0:
        _ax.scatter(defect_pos[:, 0], defect_pos[:, 1], 
                     s=150, c='#F44336', alpha=0.9, edgecolors='w', linewidths=1.0, 
                     marker='X', label=f'Defectors ({len(defect_pos)})', zorder=2)

    # --- 4. 设置图像样式和信息 ---
    _ax.set_xlim(-0.5, world_size + 0.5)
    _ax.set_ylim(-0.5, world_size + 0.5)
    _ax.set_title(f"Multi-Agent Particle System - Step: {current_steps}", fontsize=18, weight='bold')
    _ax.set_xlabel("X Coordinate", fontsize=14)
    _ax.set_ylabel("Y Coordinate", fontsize=14)
    
    # 创建图例，并调整其位置和样式
    legend = _ax.legend(loc="upper right", fontsize=12)
    legend.get_frame().set_alpha(0.8) # 设置图例背景半透明
    
    _ax.grid(True, linestyle=':', alpha=0.5)
    _ax.set_aspect('equal', adjustable='box') # 保证世界是正方形

    # --- 5. 将Matplotlib图像转换为Numpy数组 ---
    _fig.canvas.draw()
    # 从画布获取RGBA像素数据缓冲区
    rgba_buffer = _fig.canvas.buffer_rgba()
    # 将缓冲区直接转换为numpy数组
    rgb_array_with_alpha = np.asarray(rgba_buffer)
    # 取出RGB通道，忽略最后一个Alpha通道
    rgb_array = rgb_array_with_alpha[:, :, :3]
    
    return rgb_array

def display_frame(rgb_array: np.ndarray):
    """
    在 'human' 模式下，尝试在屏幕上显示一个图像帧。
    """
    global _fig, _ax
    if _fig is None or _ax is None:
        _fig, _ax = plt.subplots(figsize=(8, 8))
    
    _ax.clear()
    _ax.imshow(rgb_array)
    _ax.axis('off')
    plt.pause(0.01)

def close_render_window():
    """
    关闭由该模块创建的matplotlib窗口，以释放资源。
    """
    global _fig
    if _fig is not None:
        plt.close(_fig)
        # 重置为None，确保下次调用render时会创建新的figure
        _fig = None
        _ax = None