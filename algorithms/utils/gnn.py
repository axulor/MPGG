# gnn.py (修改后，移除实体类型 ID 依赖)

import numpy as np
# from scipy import sparse # 似乎未使用
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader # 似乎未使用
from torch_geometric.nn import MessagePassing, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops #, to_dense_batch # 似乎未使用 to_dense_batch
import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
# import torch.jit as jit # 似乎未使用
from .util import init, get_clones

"""GNN 模块定义 (已修改，移除实体类型嵌入)"""

class EmbedConv(MessagePassing):
    """
    (修改版) EmbedConv 层: 结合节点特征和可选的边特征进行消息传递。
    不再处理实体类型嵌入。
    """
    def __init__(self,
                input_dim:int,          # 输入节点特征的维度 (现在是完整维度)
                # num_embeddings:int,   # 移除
                # embedding_size:int,   # 移除
                hidden_size:int,
                layer_N:int,
                use_orthogonal:bool,
                use_ReLU:bool,
                use_layerNorm:bool,
                add_self_loop:bool,
                edge_dim:int=0):
        """
        (修改版) 初始化 EmbedConv 层。

        Args:
            input_dim (int): 节点特征维度。
            hidden_size (int): 隐藏层维度。
            layer_N (int): 隐藏线性层数量。
            use_orthogonal (bool): 是否使用正交初始化。
            use_ReLU (bool): 是否使用 ReLU。
            use_layerNorm (bool): 是否使用 LayerNorm。
            add_self_loop (bool): 是否添加自环。
            edge_dim (int, optional): 边特征维度。默认为 0。
        """
        super(EmbedConv, self).__init__(aggr='add')
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity()
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        # --- 移除实体嵌入层 ---
        # self.entity_embed = nn.Embedding(num_embeddings, embedding_size)

        # --- 修改第一个线性层的输入维度 ---
        # 输入维度 = 节点特征维度 + 边特征维度
        self.lin1 = nn.Linear(input_dim + edge_dim, hidden_size)

        # --- 后续隐藏层不变 ---
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)

        # 应用初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重。"""
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        """EmbedConv 层的前向传播。"""
        # 如果需要且没有边特征，添加自环边
        if self._add_self_loops and edge_attr is None:
            if x.size(0) > 0: # 仅在有节点时添加自环
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            # else: print("警告: EmbedConv forward 收到空节点张量，跳过自环添加。")

        # 确保 x 是 (源节点特征, 目标节点特征) 的格式
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 开始消息传递
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        """
        (修改版) 定义如何生成消息，不再处理实体类型。

        Args:
            x_j (Tensor): 源节点的特征张量，形状 [num_edges, input_dim]。
            edge_attr (OptTensor): 对应边的特征张量，形状 [num_edges, edge_dim]。

        Returns:
            Tensor: 生成的消息张量，形状 [num_edges, hidden_size]。
        """
        # --- 移除实体类型处理 ---
        # node_feat_j = x_j[:, :-1]
        # entity_type_j = x_j[:, -1].long()
        # entity_embed_j = self.entity_embed(entity_type_j)

        # --- 修改特征拼接 ---
        # 现在 x_j 就是完整的节点特征
        if edge_attr is not None:
            # 拼接节点特征和边特征
            node_feat = torch.cat([x_j, edge_attr], dim=1)
        else:
            # 只有节点特征
            node_feat = x_j

        # --- 后续线性层处理不变 ---
        x = self.lin1(node_feat) # lin1 输入维度已修改
        x = self.active_func(x)
        x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        return x


class TransformerConvNet(nn.Module):
    """
    (修改版) 基于 TransformerConv 的图卷积网络模块。
    内部的 EmbedConv 不再处理实体类型。
    """
    def __init__(self,
                input_dim:int,          # 输入节点特征维度 (不含类型ID)
                # num_embeddings:int,   # 移除
                # embedding_size:int,   # 移除
                hidden_size:int,
                num_heads:int,
                concat_heads:bool,
                layer_N:int,
                use_ReLU:bool,
                graph_aggr:str,
                global_aggr_type:str,
                embed_hidden_size:int,
                embed_layer_N:int,
                embed_use_orthogonal:bool,
                embed_use_ReLU:bool,
                embed_use_layerNorm:bool,
                embed_add_self_loop:bool,
                # max_edge_dist:float,  # 未直接使用
                edge_dim:int=1):
        """
        (修改版) 初始化 TransformerConvNet。

        Args:
            input_dim (int): 节点特征维度 (不含类型ID)。
            # ... (移除 num_embeddings, embedding_size) ...
            # ... (其他参数不变) ...
            edge_dim (int, optional): 边特征维度。默认为 1。
        """
        super(TransformerConvNet, self).__init__()
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.edge_dim = edge_dim
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type

        # --- 修改 EmbedConv 初始化 ---
        # input_dim 直接使用传入的维度 (不含ID)
        # 不再传递 num_embeddings, embedding_size
        self.embed_layer = EmbedConv(input_dim=input_dim, # 不再减 1
                            # num_embeddings=num_embeddings, # 移除
                            # embedding_size=embedding_size, # 移除
                            hidden_size=embed_hidden_size,
                            layer_N=embed_layer_N,
                            use_orthogonal=embed_use_orthogonal,
                            use_ReLU=embed_use_ReLU,
                            use_layerNorm=embed_use_layerNorm,
                            add_self_loop=embed_add_self_loop,
                            edge_dim=edge_dim)

        # --- TransformerConv 层初始化不变 ---
        # 第一个 TransformerConv 输入维度等于 EmbedConv 输出维度
        self.gnn1 = TransformerConv(in_channels=embed_hidden_size,
                                    out_channels=hidden_size,
                                    heads=num_heads,
                                    concat=concat_heads,
                                    beta=False, dropout=0.0,
                                    edge_dim=edge_dim, bias=True, root_weight=True)

        # 后续 TransformerConv 层
        self.gnn2 = nn.ModuleList()
        current_in_channels = hidden_size * num_heads if concat_heads else hidden_size
        for i in range(layer_N):
            layer = TransformerConv(in_channels=current_in_channels,
                                    out_channels=hidden_size,
                                    heads=num_heads, concat=concat_heads,
                                    beta=False, dropout=0.0,
                                    edge_dim=edge_dim, root_weight=True)
            self.gnn2.append(layer)
            current_in_channels = hidden_size * num_heads if concat_heads else hidden_size

        self.activation = nn.ReLU() if use_ReLU else nn.Tanh()

    def forward(self, batch: Union[Data, Batch]) -> Tensor:
        """TransformerConvNet 的前向传播 (逻辑不变)。"""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_indices = batch.batch

        x = self.embed_layer(x, edge_index, edge_attr) # 调用修改后的 EmbedConv
        x = self.activation(self.gnn1(x, edge_index, edge_attr))
        for gnn_layer in self.gnn2:
            x = self.activation(gnn_layer(x, edge_index, edge_attr))

        if self.graph_aggr == 'node':
            return x
        elif self.graph_aggr == 'global':
            pool_func = {'mean': global_mean_pool, 'max': global_max_pool, 'add': global_add_pool}.get(self.global_aggr_type)
            if pool_func:
                 # 确保 batch_indices 不为空且与 x 兼容
                 if batch_indices is not None and batch_indices.max() < len(torch.unique(batch_indices)): # 基础检查
                    return pool_func(x, batch_indices)
                 else: # 处理 batch_indices 无效或为空的情况
                    print(f"警告：全局池化收到无效的 batch_indices (max={batch_indices.max() if batch_indices is not None else None}, size={len(torch.unique(batch_indices)) if batch_indices is not None else None}) 或 x 为空 ({x.shape})。返回零张量。")
                    # 需要知道输出形状来创建零张量
                    num_graphs = batch_indices.max().item() + 1 if batch_indices is not None and batch_indices.numel() > 0 else 0
                    out_dim = x.shape[-1]
                    return torch.zeros((num_graphs, out_dim), device=x.device, dtype=x.dtype)
            else:
                 raise ValueError(f"不支持的全局聚合类型: {self.global_aggr_type}")
        else:
            raise ValueError(f"无效的图聚合方式: {self.graph_aggr}")


    # --- process_adj 静态方法保持不变 ---
    @staticmethod
    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        """处理邻接距离矩阵，生成 edge_index 和 edge_attr。"""
        assert adj.dim() >= 2 and adj.dim() <= 3, f"邻接矩阵维度必须是 2 或 3, 得到 {adj.dim()}"
        assert adj.size(-1) == adj.size(-2), "邻接矩阵必须是方阵"
        connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
        adj_masked = adj * connect_mask

        if adj_masked.dim() == 3:
            batch_size, num_nodes, _ = adj_masked.shape
            edge_indices_flat = adj_masked.nonzero(as_tuple=False)
            if edge_indices_flat.numel() == 0:
                 return torch.empty((2, 0), dtype=torch.int64, device=adj.device), \
                        torch.empty((0, 1), dtype=torch.float32, device=adj.device)
            edge_attr = adj_masked[edge_indices_flat[:, 0], edge_indices_flat[:, 1], edge_indices_flat[:, 2]]
            batch_offset = edge_indices_flat[:, 0] * num_nodes
            src_global = batch_offset + edge_indices_flat[:, 1]
            dst_global = batch_offset + edge_indices_flat[:, 2]
            edge_index = torch.stack([src_global, dst_global], dim=0).to(torch.int64)
        else:
            edge_index = adj_masked.nonzero(as_tuple=False).t().contiguous().to(torch.int64)
            if edge_index.numel() == 0:
                 return torch.empty((2, 0), dtype=torch.int64, device=adj.device), \
                        torch.empty((0, 1), dtype=torch.float32, device=adj.device)
            edge_attr = adj_masked[edge_index[0], edge_index[1]]

        edge_attr = edge_attr.unsqueeze(1).to(torch.float32)
        return edge_index, edge_attr



class GNNBase(nn.Module):
    """
    (修改版) 基础 GNN 包装类。
    使用修改后的 TransformerConvNet (不依赖实体嵌入)。
    """
    def __init__(self, args: argparse.Namespace,
                node_obs_dim: int,  # 单个节点特征维度 (不含类型ID)
                edge_dim: int,      # 边的特征维度
                graph_aggr: str):   # 聚合方式

        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size # GNN 内部隐藏维度 (TransformerConv 输出)
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads
        self.graph_aggr = graph_aggr # 保存本实例的聚合方式

        # --- 修改 TransformerConvNet 初始化 ---
        # input_dim 使用不含 ID 的维度
        # 不再传递 num_embeddings, embedding_size
        self.gnn = TransformerConvNet(
                    input_dim=node_obs_dim, # <--- 使用不含 ID 的维度
                    edge_dim=edge_dim,
                    # num_embeddings=args.num_embeddings, # 移除
                    # embedding_size=args.embedding_size, # 移除
                    hidden_size=args.gnn_hidden_size,
                    num_heads=args.gnn_num_heads,
                    concat_heads=args.gnn_concat_heads,
                    layer_N=args.gnn_layer_N,
                    use_ReLU=args.gnn_use_ReLU,
                    graph_aggr=graph_aggr, # 传递本实例的聚合方式
                    global_aggr_type=args.global_aggr_type,
                    embed_hidden_size=args.embed_hidden_size,
                    embed_layer_N=args.embed_layer_N,
                    embed_use_orthogonal=args.use_orthogonal,
                    embed_use_ReLU=args.embed_use_ReLU,
                    embed_use_layerNorm=args.use_feature_normalization,
                    embed_add_self_loop=args.embed_add_self_loop,
                    # max_edge_dist 参数在 TransformerConvNet 内部未使用，但 process_adj 会用
                    )

        # --- 输出维度计算保持不变 ---
        # GNN 最终输出特征的维度 (由最后一个 TransformerConv 层决定)
        self.out_dim = args.gnn_hidden_size * (args.gnn_num_heads if args.gnn_concat_heads else 1)

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor) -> Tensor:
        """
        GNNBase 的前向传播 (逻辑基本不变，但输入 node_obs 维度不同)。

        Args:
            node_obs (Tensor): 节点特征，形状 [batch_size, num_nodes, node_obs_dim (不含ID)]。
            adj (Tensor): 邻接距离矩阵，形状 [batch_size, num_nodes, num_nodes]。
            agent_id (Tensor): 需要提取特征的节点 ID，形状 [batch_size, k]。

        Returns:
            Tensor: GNN 处理后的特征表示。
        """
        batch_size, num_nodes, current_node_obs_dim = node_obs.shape
        # 验证输入维度是否与预期匹配（可选）
        # expected_node_dim = ... # 需要从初始化获取
        # assert current_node_obs_dim == expected_node_dim, f"输入 node_obs 维度 ({current_node_obs_dim}) 与预期 ({expected_node_dim}) 不符"

        # 1. 转换 adj 为 edge_index, edge_attr
        edge_index, edge_attr = TransformerConvNet.process_adj(adj, self.args.max_edge_dist)

        # 2. 准备 PyG Batch 数据
        x = node_obs.reshape(-1, current_node_obs_dim) # 展平
        batch_indices = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_indices)

        # 3. 通过核心 GNN 网络
        # 输出 x_processed 形状:
        #   - 'node': [batch_size * num_nodes, self.out_dim]
        #   - 'global': [batch_size, self.out_dim]
        x_processed = self.gnn(data)

        # 4. 根据聚合方式处理输出
        if self.graph_aggr == 'node':
            # 需要提取特定节点的特征
            x_reshaped = x_processed.view(batch_size, num_nodes, self.out_dim)
            agent_id = agent_id.long()
            k = agent_id.shape[1]
            # 使用 gather 提取
            idx_expanded = agent_id.unsqueeze(-1).expand(-1, k, self.out_dim)
            gathered_features = x_reshaped.gather(1, idx_expanded) # [batch_size, k, self.out_dim]
            # 拼接成 [batch_size, k * self.out_dim]
            output_features = gathered_features.view(batch_size, -1)
            return output_features
        elif self.graph_aggr == 'global':
            # 已经是全局聚合后的结果 [batch_size, self.out_dim]
            return x_processed
        else:
            raise ValueError(f"GNNBase 不支持的聚合方式: {self.graph_aggr}")