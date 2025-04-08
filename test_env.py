# from pettingzoo.test import parallel_api_test
# from envs.migratory_pgg_env_v4 import MigratoryPGGEnv

# if __name__ == "__main__":
#     env = MigratoryPGGEnv()
#     parallel_api_test(env, num_cycles=10000)  # 可调大一点做 stress 测试

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------- 模拟 GNNEncoder 模块 --------
class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x[0]  # 只取中心节点的 embedding

# -------- 构造一个图观测（中心 + 3邻居） --------
# 中心智能体特征
self_features = torch.tensor([[1.0, 0.5, 0.0, 0.0, 0.0]])  # shape: [1, 5]

# 三个邻居特征
neighbor_features = torch.tensor([
    [0.0, 1.2, 0.1, 0.3, 0.4],
    [1.0, 0.8, 0.2, -0.1, 0.6],
    [0.0, 0.2, -0.5, 0.4, 0.7]
])  # shape: [3, 5]

# 拼接节点特征
x = torch.cat([self_features, neighbor_features], dim=0)  # shape: [4, 5]

# 构建边（所有邻居 → 中心）
edge_index = torch.tensor([
    [1, 2, 3],  # from neighbors
    [0, 0, 0]   # to self
], dtype=torch.long)

# 构造图数据
data = Data(x=x, edge_index=edge_index)

# -------- 实际运行 GNN --------
model = SimpleGNN()
output = model(data)

# 打印输出
print("Final encoded observation (Box(64,)):\n", output)
print("Shape:", output.shape)
