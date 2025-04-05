# ———————————————————— 训练结束后的可视化 ———————————————————————— #
# 计算每个网格的平均合作率（取最后1000步的合作率进行平均）
grid_avg_coop = {}
for grid, coop_list in cooperation_history.items():
    if len(coop_list) > 0:
        # 取最后 1000 步（或所有步数，如果不足1000步）的平均值
        last_data = coop_list[-20000:]
        grid_avg_coop[grid] = np.mean(last_data)
    else:
        grid_avg_coop[grid] = None

# 构造热力图数据：二维数组，行列分别为 grid_division，注意上下翻转以使得纵轴从下往上增大
heatmap_data = np.flipud(np.zeros((env.grid_division, env.grid_division)))
for (grid_row, grid_col), coop_rate in grid_avg_coop.items():
    heatmap_data[grid_row, grid_col] = coop_rate if coop_rate is not None else np.nan

# 绘制热力图
plt.figure(figsize=(8, 6))
ax = sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Average Cooperation Rate'})
ax.invert_yaxis()  # 反转 y 轴，使得标签从下往上增大
plt.title("Heatmap of Average Cooperation Rate per Grid (Last 1000 Steps)")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.show()


# ———————————————————— 绘制指定网格合作率随时间变化的曲线 ———————————————————————— #
# 定义滑动窗口函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 指定三个网格
specified_grids = [(3, 4), (5, 2), (0, 0)]
window_size = 1000  # 平滑窗口大小

plt.figure(figsize=(12, 6))

# 对于每个指定网格，计算完整时间步下（300k步）的合作率平滑曲线
for grid in specified_grids:
    # 从合作率记录中获取该网格的原始时间序列数据（长度应该为训练的步数）
    coop_time_series = np.array(cooperation_history.get(grid, []))
    
    # 计算滑动平均得到平滑后的曲线
    coop_smoothed = moving_average(coop_time_series, window_size)
    
    # 构造横轴（平滑后数据的时间步索引，注意长度会减少 window_size - 1）
    time_steps = np.arange(len(coop_smoothed))
    
    plt.plot(time_steps, coop_smoothed, label=f"Grid {grid}")

plt.xlabel("Time Step")
plt.ylabel("Smoothed Cooperation Rate")
plt.title(f"Cooperation Rate Over Time for Specified Grids (Smoothed over {window_size} Steps)")
plt.legend()
plt.show()

