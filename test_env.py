import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

def plot_agent_connections(num_agents: int, 
                            world_size: float, 
                            radius: float, 
                            dot_size: int = 50, 
                            seed: int = None):
    """
    绘制两个快照，并正确处理周期性边界的连线：
    1. 普通邻接图：智能体之间距离在 radius 内则连线。
    2. 超边结构示意图：从每个智能体向其 radius 内的其他智能体连线。
    """
    if seed is not None:
        np.random.seed(seed)

    positions = np.random.uniform(0, world_size, (num_agents, 2))

    # --- Helper function to plot lines with periodic boundary conditions ---
    def plot_periodic_line(ax, p1, p2, world_size, color='k', alpha=0.3, linewidth=0.8, zorder=1):
        """
        绘制两点之间的连线，考虑周期性边界。
        p1, p2: 形状为 (2,) 的数组，代表点的坐标 [x, y]
        """
        dx_raw = p2[0] - p1[0]
        dy_raw = p2[1] - p1[1]

        # 计算考虑周期性边界的最短差值分量
        dx_periodic = (dx_raw + world_size / 2) % world_size - world_size / 2
        dy_periodic = (dy_raw + world_size / 2) % world_size - world_size / 2

        # 检查是否跨越了x边界
        if abs(dx_raw) > world_size / 2: # 实际的x差值大于L/2，说明最短路径跨边界
            # 画两段线
            # 第一段：从 p1 到 x 边界
            x_mid1_p1 = world_size if dx_raw < 0 else 0 # p1 接触的 x 边界 (如果p2在p1右边且跨界, p1到右边界L; 如果p2在p1左边且跨界, p1到左边界0)
            y_mid1_p1 = p1[1] + dy_periodic * (abs(x_mid1_p1 - p1[0]) / abs(dx_periodic)) if abs(dx_periodic) > 1e-7 else p1[1]
            
            ax.plot([p1[0], x_mid1_p1], [p1[1], y_mid1_p1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
            
            # 第二段：从另一侧 x 边界到 p2
            x_mid1_p2 = 0 if dx_raw < 0 else world_size # p2 接触的 x 边界 (与上面相反)
            y_mid1_p2 = p2[1] - dy_periodic * (abs(p2[0] - x_mid1_p2) / abs(dx_periodic)) if abs(dx_periodic) > 1e-7 else p2[1]
            ax.plot([x_mid1_p2, p2[0]], [y_mid1_p2, p2[1]], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        else: # x方向不跨界
            dx_to_plot = dx_raw # 直接使用原始x差值

        # 检查是否跨越了y边界
        if abs(dy_raw) > world_size / 2:
            # 画两段线
            y_mid2_p1 = world_size if dy_raw < 0 else 0
            x_mid2_p1 = p1[0] + dx_periodic * (abs(y_mid2_p1 - p1[1]) / abs(dy_periodic)) if abs(dy_periodic) > 1e-7 else p1[0]
            ax.plot([p1[0], x_mid2_p1], [p1[1], y_mid2_p1], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

            y_mid2_p2 = 0 if dy_raw < 0 else world_size
            x_mid2_p2 = p2[0] - dx_periodic * (abs(p2[1] - y_mid2_p2) / abs(dy_periodic)) if abs(dy_periodic) > 1e-7 else p2[0]
            ax.plot([x_mid2_p2, p2[0]], [y_mid2_p2, p2[1]], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        else: # y方向不跨界
            dy_to_plot = dy_raw

        # 如果x和y方向都不跨界，则直接画一条直线
        if abs(dx_raw) <= world_size / 2 and abs(dy_raw) <= world_size / 2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        # 注意：如果同时跨越x和y边界，上面的逻辑会画出四条线段的近似（两条x跨界，两条y跨界），
        # 更精确的画法是找到对角线穿过的点，但这会更复杂。
        # 目前的简化逻辑是，如果x跨界，就画x跨界的两条线；如果y跨界，就画y跨界的两条线。
        # 如果只想画最短路径的那一条（或两条）线，逻辑需要调整。

    # --- 为了简化并直接使用最短矢量来画线 ---
    # 我们重新定义一个画线函数，它接受原始点和最短矢量
    def plot_shortest_path_line(ax, p1_coords, p2_coords, world_size, color, alpha, linewidth, zorder):
        """Plots a line between p1 and p2, handling periodic boundaries by drawing segments."""
        x1, y1 = p1_coords
        x2, y2 = p2_coords

        dx = x2 - x1
        dy = y2 - y1

        # Apply periodic boundary correction to find the shortest vector components
        if abs(dx) > world_size / 2:
            dx = dx - np.sign(dx) * world_size
        if abs(dy) > world_size / 2:
            dy = dy - np.sign(dy) * world_size
        
        # Target point for the shortest path line segment from p1
        x2_eff = x1 + dx
        y2_eff = y1 + dy

        # Now, draw segments if x2_eff or y2_eff are outside [0, world_size] due to dx, dy
        # This means the line segment (x1,y1) to (x2_eff,y2_eff) might cross boundaries
        
        points_to_plot_x = [x1]
        points_to_plot_y = [y1]

        # Handle X wrap-around
        if x2_eff > world_size: # Wrapped from left to right
            points_to_plot_x.extend([world_size, 0])
        elif x2_eff < 0: # Wrapped from right to left
            points_to_plot_x.extend([0, world_size])
        points_to_plot_x.append(x2_eff % world_size) # Ensure final x is within bounds

        # Handle Y wrap-around
        if y2_eff > world_size: # Wrapped from bottom to top
            points_to_plot_y.extend([world_size, 0])
        elif y2_eff < 0: # Wrapped from top to bottom
            points_to_plot_y.extend([0, world_size])
        points_to_plot_y.append(y2_eff % world_size) # Ensure final y is within bounds
        
        # This logic is still tricky for plotting.
        # A common way is to plot multiple "ghost" copies of the world.
        # For simplicity here, we'll use the shortest vector and plot segments.

        # Simplified approach: plot lines to 8 neighbors + original
        # This is often easier than complex line segmenting for visualization
        for i_offset in [-1, 0, 1]:
            for j_offset in [-1, 0, 1]:
                if i_offset == 0 and j_offset == 0:
                    p2_draw = p2_coords # Original p2
                else:
                    # p2 in a neighboring "ghost" cell
                    p2_draw = (p2_coords[0] + i_offset * world_size, 
                               p2_coords[1] + j_offset * world_size) 
                
                # Calculate distance to this potentially shifted p2
                current_dx = p2_draw[0] - p1_coords[0]
                current_dy = p2_draw[1] - p1_coords[1]
                dist_sq = current_dx**2 + current_dy**2

                if dist_sq <= radius**2 + 1e-9: # Check distance to this image of p2
                                                # (add small epsilon for float comparisons)
                    ax.plot([p1_coords[0], p2_draw[0]], [p1_coords[1], p2_draw[1]],
                            color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
                    # We only need to draw one shortest line.
                    # If we found a connection to a ghost, we are done for this pair.
                    # However, for the "Agent-Centered Group Connections" (图2), we draw from i to all its neighbors.
                    # So, for图2, this "draw once" logic is per (i,j) pair.
                    # For 图1 (Pairwise), we only need one line between i and j.

    # Recalculate dist_matrix without periodic correction for raw differences later
    # delta_raw = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # For shortest distance, we use the already corrected dist_matrix from the MIC
    
    # Recompute dist_matrix, this time it's the actual shortest distance due to MIC
    delta_mic = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    delta_mic = (delta_mic + world_size / 2) % world_size - world_size / 2
    dist_matrix_mic = np.linalg.norm(delta_mic, axis=2)


    # --- 图1：普通邻接图 (使用修正后的画线逻辑) ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(positions[:, 0], positions[:, 1], s=dot_size, c='blue', alpha=0.7, zorder=10) # zorder higher

    for i, j in combinations(range(num_agents), 2):
        if dist_matrix_mic[i, j] <= radius:
            # plot_periodic_line(ax1, positions[i], positions[j], world_size, color='k', alpha=0.3, linewidth=0.8, zorder=1)
            # Use the simpler ghosting method for plotting lines across boundaries
            plot_shortest_path_line_simplified(ax1, positions[i], positions[j], world_size, radius, 'k', 0.3, 0.8, 1)


    ax1.set_xlim(0, world_size)
    ax1.set_ylim(0, world_size)
    ax1.set_xlabel(f"X (World Size: {world_size})", fontsize=12)
    ax1.set_ylabel(f"Y (World Size: {world_size})", fontsize=12)
    ax1.set_title(f"Pairwise Adjacency Graph (N={num_agents}, R={radius}) - Periodic", fontsize=14)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 图2：突出显示以每个智能体为中心的“超边”结构 (使用修正后的画线逻辑) ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(positions[:, 0], positions[:, 1], s=dot_size, c='red', alpha=0.7, zorder=10)

    for i in range(num_agents):
        for j in range(num_agents): # Iterate all, including self, but dist will be 0
            if i == j: continue 
            if dist_matrix_mic[i, j] <= radius:
                # plot_periodic_line(ax2, positions[i], positions[j], world_size, color='b', alpha=0.2, linewidth=1.0, zorder=1)
                plot_shortest_path_line_simplified(ax2, positions[i], positions[j], world_size, radius, 'b', 0.2, 1.0, 1)


    ax2.set_xlim(0, world_size)
    ax2.set_ylim(0, world_size)
    ax2.set_xlabel(f"X (World Size: {world_size})", fontsize=12)
    ax2.set_ylabel(f"Y (World Size: {world_size})", fontsize=12)
    ax2.set_title(f"Agent-Centered Connections (N={num_agents}, R={radius}) - Periodic", fontsize=14)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.show()

# Helper function for simplified periodic line plotting
def plot_shortest_path_line_simplified(ax, p1_coords, p2_coords, world_size, radius_check, color, alpha, linewidth, zorder):
    """
    Plots the shortest line between p1 and p2 considering periodic boundaries.
    It iterates through the 8 neighboring "ghost" cells for p2 and p1 itself.
    """
    min_dist_sq = float('inf')
    best_p2_draw = p2_coords # Default to original p2

    # Check original and 8 ghost images of p2 relative to p1
    for dx_offset_factor in [-1, 0, 1]:
        for dy_offset_factor in [-1, 0, 1]:
            p2_candidate = (p2_coords[0] + dx_offset_factor * world_size,
                            p2_coords[1] + dy_offset_factor * world_size)
            
            dist_sq = (p1_coords[0] - p2_candidate[0])**2 + (p1_coords[1] - p2_candidate[1])**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_p2_draw = p2_candidate
    
    # Only draw if the shortest distance is within the radius
    if min_dist_sq <= radius_check**2 + 1e-9: # Add epsilon for float comparison
        ax.plot([p1_coords[0], best_p2_draw[0]], [p1_coords[1], best_p2_draw[1]],
                color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


if __name__ == '__main__':
    NUM_AGENTS = 25
    WORLD_SIZE = 7.0
    RADIUS = 2.0
    DOT_SIZE = 70
    RANDOM_SEED = 42

    plot_agent_connections(NUM_AGENTS, WORLD_SIZE, RADIUS, DOT_SIZE, RANDOM_SEED)
    
    # Example with more obvious wrapping
    # plot_agent_connections(num_agents=5, world_size=10.0, radius=3.0, dot_size=100, seed=10) 
    # For seed=10 with N=5, R=3:
    # Agent positions might be:
    # [[8.6617615 , 4.50516897],
    #  [0.26092142, 1.847359 pemimpin],
    #  [9.74587496, 9.4095982 ],
    #  [9.73307345, 2.4247674 ],
    #  [0.10933638, 3.663526  ]]
    # Agent 0 (8.6, 4.5) and Agent 1 (0.2, 1.8) should connect across boundary. dx_raw = -8.4, dx_periodic = 1.6