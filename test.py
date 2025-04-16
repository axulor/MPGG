import torch

# 假设我们有 2 个图，每个图 3 个节点，每个节点是 4 维特征
x = torch.tensor([
    [   # 图0
        [1, 2, 3, 4],    # 节点0
        [5, 6, 7, 8],    # 节点1
        [9, 10, 11, 12]  # 节点2
    ],
    [   # 图1
        [13, 14, 15, 16],  # 节点0
        [17, 18, 19, 20],  # 节点1
        [21, 22, 23, 24]   # 节点2
    ]
])  # shape = [2, 3, 4] → 2个图、每图3节点、每节点4维

# 假设我们想要图0的第 2 和 图1的 第 0 个节点的特征
agent_id = torch.tensor([[2], [0]])  # shape = [2, 1]

# 核心操作：从 x 中提取每个图指定节点的向量
result = x.gather(
    1,
    agent_id.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [2, 1, 4]
).squeeze(1)  # [2, 4]

# 打印结果
print("原始 x:")
print(x)
print("\nagent_id:")
print(agent_id)
print(agent_id.unsqueeze(-1))
print(x.size(-1))
print(agent_id.unsqueeze(-1).expand(-1, -1, x.size(-1)))
print("\n提取后的结果:")
print(result)
