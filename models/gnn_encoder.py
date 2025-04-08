# gnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 PyTorch Geometric 导入数据类和图卷积层
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def obs_to_pyg_data(obs: dict) -> Data:
    """
    将 PettingZoo 环境产生的单个智能体的观测字典转换为 PyG 的 Data 对象。
    
    输入:
        obs: 观测字典，格式如下：
            {
                "self": {"strategy": int, "last_payoff": float},
                "neighbors": [
                    {"relative_position": np.array(shape=(2,)), "distance": float,
                     "strategy": int, "last_payoff": float},
                    ...
                ]
            }
    
    输出:
        data: PyG 的 Data 对象，其中：
            - x: 节点特征矩阵，第一行为中心节点（当前智能体）的特征，
                 后续行为每个邻居节点的特征。所有节点均拥有相同维度（本例中设为5）。
            - edge_index: 边索引张量（bidirectional star graph），中心与各邻居双向连接。
    """
    # 为了保证中心节点和邻居节点的特征维度一致，我们设计节点特征维度为5：
    # 对于邻居节点，特征构成为：
    # [strategy, last_payoff, relative_position[0], relative_position[1], distance]
    #
    # 而中心节点（当前智能体）的观测仅包含 strategy 和 last_payoff，
    # 因此我们采用填0的方式补充相应缺失的特征，构成：
    # [strategy, last_payoff, 0.0, 0.0, 0.0]
    
    # 中心节点特征
    center_feature = [
        float(obs["self"]["strategy"]),
        float(obs["self"]["last_payoff"]),
        0.0, 0.0, 0.0  # padding for relative_position 和 distance
    ]
    
    # 收集邻居节点特征
    neighbor_features = []
    for neighbor in obs.get("neighbors", []):
        feat = [
            float(neighbor["strategy"]),
            float(neighbor["last_payoff"])
        ]
        rel_pos = neighbor["relative_position"]
        # 增加2维 relative_position 特征
        feat.extend([float(rel_pos[0]), float(rel_pos[1])])
        # 增加 distance 特征
        feat.append(float(neighbor["distance"]))
        neighbor_features.append(feat)
    
    # 将中心节点和邻居节点特征整合成节点特征矩阵
    x = torch.tensor([center_feature] + neighbor_features, dtype=torch.float)
    
    num_nodes = x.shape[0]
    # 构造 star graph 的边：中心节点的下标固定为0，与每个邻居双向相连
    if num_nodes > 1:
        # 从中心节点(0)到每个邻居节点（下标1到 num_nodes-1）
        src = [0] * (num_nodes - 1) + list(range(1, num_nodes))
        dst = list(range(1, num_nodes)) + [0] * (num_nodes - 1)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # 若只有中心节点，则无边
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 构建 PyG Data 对象
    data = Data(x=x, edge_index=edge_index)
    return data

class GNNEncoder(nn.Module):
    """
    图神经网络编码器，用于从 PyG 的 Data 对象中提取固定维度的特征向量。
    采用两层 GCN：第一层后使用 ReLU 激活，第二层输出最终特征。
    
    在本设计中，由于每个图为一个 star graph，
    我们直接采用中心节点（下标 0）的特征作为整个图的表示。
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        参数:
            in_channels: 节点输入特征维数（本例为5）
            hidden_channels: 第1层 GCN 的输出特征维数
            out_channels: 第2层 GCN 的输出特征维数（最终编码后的向量维数）
        """
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播:
            - data: PyG Data 对象
            - 先经过两层 GCN 得到节点特征，再选择中心节点的输出作为最终表示
        
        返回:
            一个 shape 为 [out_channels] 的向量
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # 直接取中心节点（下标 0）的特征作为图表示
        return x[0]
        
# # 测试示例（可选）
# if __name__ == "__main__":
#     # 构造一个模拟观测数据
#     sample_obs = {
#         "self": {"strategy": 1, "last_payoff": 0.5},
#         "neighbors": [
#             {
#                 "relative_position": [0.707, 0.707],
#                 "distance": 1.0,
#                 "strategy": 0,
#                 "last_payoff": 0.2
#             },
#             {
#                 "relative_position": [-0.5, 0.866],
#                 "distance": 1.2,
#                 "strategy": 1,
#                 "last_payoff": -0.1
#             },
#             {
#                 "relative_position": [0.607, 0.707],
#                 "distance": 2.0,
#                 "strategy": 1,
#                 "last_payoff": 0.8
#             }
#         ]
#     }
    
#     # 转换为 PyG Data 对象
#     data = obs_to_pyg_data(sample_obs)
#     print("节点特征矩阵 x:")
#     print(data.x)
#     print("边索引 edge_index:")
#     print(data.edge_index)
    
#     # 创建编码器实例，输入特征维数为5，隐藏层维数设为16，输出维数设为64
#     encoder = GNNEncoder(in_channels=5, hidden_channels=16, out_channels=64)
#     # 前向传播得到编码后的向量
#     output = encoder(data)
#     print("编码后的向量 shape:", output.shape)
