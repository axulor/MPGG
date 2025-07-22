import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, TransformerConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops 
import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from .util import init, get_clones, check

"""GNN 模块定义 """

class EmbedConv(MessagePassing):
    """
    EmbedConv 层: 对节点特征和可选的边特征进行初步变换

    """
    def __init__(self,
                input_dim: int,      # 输入节点特征的维度
                hidden_size: int,    # EmbedConv 内部 MLP 的隐藏维度
                layer_N: int,        # EmbedConv 内部 MLP 的层数
                use_orthogonal: bool,
                use_ReLU: bool,
                use_layerNorm: bool, # 是否在 EmbedConv 的 MLP 中使用 LayerNorm
                add_self_loop: bool, # 是否为消息传递添加自环 
                edge_dim: int,       # 输入的边特征维度
                device=torch.device("cpu")
            ):
        
        super(EmbedConv, self).__init__(aggr='add') # 聚合方式 'add' 是消息传递的默认方式之一

        self.add_self_loop = add_self_loop

        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh() # 选择激活函数
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity() # 选择是否层归一化
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_ # 选择权重初始化方法

        # 输入层，输入维度 = 节点特征维度 + 边特征维度
        self.lin1 = nn.Linear(input_dim + edge_dim, hidden_size) 
        
        # 后续隐藏层
        self.layers = nn.ModuleList()
        for _ in range(layer_N): 
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            if use_layerNorm: 
                self.layers.append(nn.LayerNorm(hidden_size)) # 注意这里不能重复添加同一个 self.layer_norm

        self.initialize_weights() # 初始化所有定义的线性层的权重和偏置

        self.to(device)

    def initialize_weights(self):
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain) 
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                    edge_attr: OptTensor = None)-> Tensor:
        """EmbedConv 层的前向传播。"""
        # 对于孤立节点边特征，添加自环边
        if self.add_self_loop and edge_attr is None:
            if x.size(0) > 0: # 仅在有节点时添加自环
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 确保 x 是 (源节点特征, 目标节点特征) 的格式
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # 开始消息传递
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        """ 
        self.propagate() 的调用过程中被隐式执行
        - x_j: 节点的特征 [num_edges, input_dim]
        - edge_attr: 对应边的特征 [num_edges, edge_dim]
        """
        
        # 拼接源节点特征和边特征作为消息输入
        message_input = torch.cat([x_j, edge_attr], dim=1) # [num_edges, input_dim + edge_dim]

        # 通过第一个输入层
        x = self.lin1(message_input)
        x = self.active_func(x)
        if not isinstance(self.layer_norm, nn.Identity): # 只有当 layer_norm 不是 Identity 时才应用
            x = self.layer_norm(x) # 第一个MLP块后的LayerNorm

        # 通过后续的隐藏层
        for layer in self.layers:
            x = layer(x)

        return x
    

class TransformerConvNet(nn.Module):
    def __init__(self,
                input_dim:int,      # 节点特征维度
                edge_dim:int,       # 边特征维度
                hidden_size:int,    # GNN 隐藏层特征维度
                num_heads:int,      # GNN 注意力机制头数
                concat_heads:bool,  # 是否拼接头数
                layer_N:int,        # GNN卷积层数量
                use_ReLU:bool,      # 是否使用 ReLU 激活函数
                # EmbedConv 参数 
                embed_hidden_size:int, 
                embed_layer_N:int,
                embed_use_orthogonal:bool,
                embed_use_ReLU:bool,
                embed_use_layerNorm:bool,
                embed_add_self_loop:bool, # 是否添加自环以聚合自身信息
                # 控制最终输出
                graph_aggr:str,         # "node" 或 "global"
                global_aggr_type:str,    # "mean", "max", "add" (仅当 graph_aggr="global")
                device=torch.device("cpu")
                ):
        super(TransformerConvNet, self).__init__()
        
        # 基本属性
        self.graph_aggr = graph_aggr # 最终聚合方式
        self.global_aggr_type = global_aggr_type # 图级聚合方式
        self.activation = nn.ReLU() if use_ReLU else nn.Tanh() # 激活函数

        # EmbedConv 层 self.embed_layer
        self.embed_layer = EmbedConv( # 输出维度是 embed_hidden_size
            input_dim=input_dim, 
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop, 
            edge_dim=edge_dim
        )

        # TransformerConv 层 self.gnn_conv_layers
        self.gnn_conv_layers = nn.ModuleList() # 创建TransformerConv 层列表
        size_in_channels = embed_hidden_size # 初始输入维度为 EmbedConv 层输出维度
        for _ in range(layer_N ): # 构建 TransformerConv 层
            layer = TransformerConv(
                in_channels=size_in_channels,
                out_channels=hidden_size,
                heads=num_heads,
                concat=concat_heads,
                beta=False,
                dropout=0.0,
                edge_dim=edge_dim,
                bias=True,
                root_weight=True # 允许层学习一个权重给中心节点自身的特征
            )
            self.gnn_conv_layers.append(layer) # 堆叠 TransformerConv 层
            size_in_channels = hidden_size * num_heads if concat_heads else hidden_size # 更新层间维度
        
        # TransformerConvNet 的最终输出特征维度 (在池化之前)
        self.final_node_embedding_dim = size_in_channels

        self.to(device)

    def forward(self, data: Union[Data, Batch]) -> Tensor:
        """
        Return:
        - x_final_nodes: 
        形状 (M*N, self.final_node_embedding_dim) 或者 (M, self.final_node_embedding_dim)
        """

        # 解包 pyg_batch_data 
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch

        # 初始嵌入层
        x_embedded = self.embed_layer(x, edge_index, edge_attr) # 形状 [M*N, embed_hidden_size]

        # 通过后续的 TransformerConv 层
        x_processed = x_embedded # 从 embed_layer 的输出开始
        for layer in self.gnn_conv_layers:
            # layer 的输入是上一层处理的结果 x_processed
            x_processed = self.activation(layer(x_processed, edge_index, edge_attr=edge_attr))
        
        x_final_nodes = x_processed  # 形状 [M*N, self.final_node_embedding_dim]

        # 根据聚合方式决定最终输出
        if self.graph_aggr == 'node':
            # 返回所有节点的最终嵌入
            return x_final_nodes # 形状 (M*N, self.final_node_embedding_dim)     
        elif self.graph_aggr == 'global':
            # 根据全局图聚合方式处理输出
            pool_func = {'mean': global_mean_pool,
                        'max': global_max_pool,
                        'add': global_add_pool}.get(self.global_aggr_type)
            if pool_func:
                return pool_func(x_final_nodes, batch_index) # 形状: (M, self.final_node_embedding_dim)
            else:
                raise ValueError(f"不支持的全局图聚合类型: {self.global_aggr_type}")
        else:
            raise ValueError(f"无效的图聚合方式: {self.graph_aggr}")


    @staticmethod
    def process_adj(adj_batch: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        """
        Ruturn:
        - edge_index_batched: 形状 [2, Num_Total_Edges_In_Batch]
        - edge_attr_batched: 形状 [Num_Edges, 1] (边属性是标量距离)
        """
        assert adj_batch.dim() == 3, f"process_adj 期望3维的 adj_batch (M,N,N), 得到 {adj_batch.dim()}"
        assert adj_batch.size(1) == adj_batch.size(2), "每个邻接矩阵必须是方阵"
        
        # 1. 创建连接掩码并转换为浮点数 (0.0 或 1.0)
        connect_mask = ((adj_batch < max_edge_dist) & (adj_batch > 0)).float()
        adj_masked = adj_batch * connect_mask # 形状 (M, N, N), 只保留有效边的距离值

        # 2. 找到所有非零元素的索引，以元组形式返回 (b_indices, r_indices, c_indices)
        indices_tuple = adj_masked.nonzero(as_tuple=True) 
        # indices_tuple[0] 是批次索引 (b), indices_tuple[1] 是行索引 (r), indices_tuple[2] 是列索引 (c)
        # 这三个张量的形状都是 (Num_Total_Edges_In_Batch,)

        # 3. 检查是否有边
        if indices_tuple[0].numel() == 0: # 检查第一个索引张量的元素数量是否为0
            return torch.empty((2, 0), dtype=torch.long, device=adj_batch.device), \
                torch.empty((0, 1), dtype=torch.float32, device=adj_batch.device)
        
        # 4. 直接使用元组中的索引来获取边属性
        #    adj_masked[indices_tuple[0], indices_tuple[1], indices_tuple[2]]
        #    等效于 adj_masked[b_indices, r_indices, c_indices]
        edge_attr_flat = adj_masked[indices_tuple] # 形状 (Num_Total_Edges_In_Batch,)
        
        # 5. 构建全局 edge_index
        _ , num_nodes, _ = adj_batch.shape
        batch_idx_of_edges = indices_tuple[0] # 非零边所在的图的索引
        src_in_graph = indices_tuple[1]       # 非零边在各自图内的源节点索引
        dst_in_graph = indices_tuple[2]       # 非零边在各自图内的目标节点索引

        batch_offset = batch_idx_of_edges * num_nodes 
        src_global = batch_offset + src_in_graph
        dst_global = batch_offset + dst_in_graph
        edge_index_batched = torch.stack([src_global, dst_global], dim=0).to(torch.long)

        # 6. 统一 edge_attr 的形状
        edge_attr_batched = edge_attr_flat.unsqueeze(1).to(torch.float32)
        
        return edge_index_batched, edge_attr_batched



class GNNBase(nn.Module):
    def __init__(self, 
                args: argparse.Namespace,
                node_obs_dim: int,  # 单个节点特征的维度, MPGG 中为 6
                edge_dim: int,      # 边的特征维度, MPGG 中为7
                graph_aggr: str,    # 聚合方式, "node" 或 "global"
                use_mini_node_feat: bool = False,
                device=torch.device("cpu")
                ):
        super(GNNBase, self).__init__()

        self.args = args
        self.graph_aggr = graph_aggr
        self.edge_dim = edge_dim 
        self.use_mini_node_feat = use_mini_node_feat
        self.k_neighbors = args.k_neighbors

        # 实例化核心的 GNN 网络 (TransformerConvNet)
        self.gnn_core = TransformerConvNet( # 重命名为 gnn_core 以示区分
            input_dim=node_obs_dim,
            edge_dim=self.edge_dim,
            # TransformerConvNet 内部使用的参数
            hidden_size=args.gnn_hidden_size, 
            num_heads=args.gnn_num_heads,
            concat_heads=args.gnn_concat_heads,
            layer_N=args.gnn_layer_N, 
            use_ReLU=args.gnn_use_ReLU, 
            # EmbedConv 相关的参数 
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal, 
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feature_normalization, 
            embed_add_self_loop=args.embed_add_self_loop,
            # 参数控制 TransformerConvNet 
            graph_aggr=graph_aggr, 
            global_aggr_type=args.global_aggr_type, 
        )

        # 定义 GNNBase 的输出维度 self.out_dim
        if args.gnn_concat_heads: # 启用头拼接
            out_dim = args.gnn_hidden_size * args.gnn_num_heads
        else:
            out_dim = args.gnn_hidden_size
        
        self.out_dim = out_dim 

        self.to(device)

    def _process_graph_data(self, node_obs: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        M, N, _ = node_obs.shape
        device = node_obs.device

        if self.k_neighbors <= 0:
            return self._process_graph_data_original(node_obs, adj)

        distances = adj.clone()
        mask = (adj == 0) | (adj > self.args.radius)
        distances[mask] = float('inf')

        top_k_dists, top_k_indices = torch.topk(distances, k=self.k_neighbors, dim=-1, largest=False)

        valid_edge_mask = (top_k_dists != float('inf'))

        src_local_indices = torch.arange(N, device=device).view(1, N, 1).expand(M, N, self.k_neighbors)

        batch_indices = torch.arange(M, device=device).view(M, 1, 1).expand(M, N, self.k_neighbors)

        b = batch_indices[valid_edge_mask]
        src_local = src_local_indices[valid_edge_mask]
        dst_local = top_k_indices[valid_edge_mask]

        if b.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), \
                torch.empty((0, self.edge_dim), dtype=torch.float32, device=device)

        offset = b * N
        src_global = src_local + offset
        dst_global = dst_local + offset
        edge_index = torch.stack([src_global, dst_global], dim=0)

        positions = node_obs[..., 0:2] * self.args.world_size
        rel_pos = (positions[b, dst_local] - positions[b, src_local]) / self.args.radius

        strategies = node_obs[..., 4].long()
        src_str, dst_str = strategies[b, src_local], strategies[b, dst_local]
        interaction_type = src_str * 2 + dst_str
        strat_one_hot = torch.nn.functional.one_hot(interaction_type, num_classes=4).float()

        distance = adj[b, src_local, dst_local].unsqueeze(-1)
        distance_normalized = distance / self.args.radius

        payoffs = node_obs[..., 5] 
    
        payoff_i = payoffs[b, src_local] 
        payoff_j = payoffs[b, dst_local]  
        
        payoff_difference = payoff_j - payoff_i

        normalized_payoff_diff = torch.tanh(payoff_difference).unsqueeze(-1)

        edge_attr = torch.cat([rel_pos, strat_one_hot, distance_normalized,normalized_payoff_diff], dim=-1)
        
        return edge_index, edge_attr

    def _process_graph_data_original(self, node_obs: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes batch of node features and adjacency matrices
        to create pyg-compatible edge_index and a rich edge_attr.
        """
        M, N, D_obs = node_obs.shape
        
        # 1. Find all existing edges from the adjacency matrix
        connect_mask = ((adj > 0) & (adj <= self.args.max_edge_dist))
        b, dst, src = connect_mask.nonzero(as_tuple=True)
        # b: batch_idx, dst: destination_idx (row), src: source_idx (col)
        # Note: PyG convention is (source, destination)
        
        if b.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=node_obs.device), \
                torch.empty((0, self.edge_dim), dtype=torch.float32, device=node_obs.device)
        
        # 2. Build edge_index in pyg format (source_to_destination)
        batch_offset = b * N
        src_global = batch_offset + src
        dst_global = batch_offset + dst
        edge_index = torch.stack([src_global, dst_global], dim=0)

        # 3. Build the rich edge_attr tensor (NEW)
        
        # a. Relative position (source -> destination)
        # pos is in obs[..., 0:2]. We normalize by world_size.
        positions = node_obs[..., 0:2] * self.args.world_size
        # Get positions of all source and destination nodes of the edges
        rel_pos = (positions[b, dst] - positions[b, src]) / self.args.radius

        # b. Strategy interaction type (one-hot encoded)
        # strategy is in obs[..., 4]
        strategies = node_obs[..., 4].long() # Shape (M, N)
        src_str, dst_str = strategies[b, src], strategies[b, dst]
        # Interaction type: C->C=3, C->D=2, D->C=1, D->D=0
        interaction_type = src_str * 2 + dst_str 
        strat_one_hot = torch.nn.functional.one_hot(interaction_type, num_classes=4).float() # (Num_Edges, 4)

        # c. Distance (scalar)
        distance = adj[b, dst, src].unsqueeze(-1) # (Num_Edges, 1)
        distance_normalized = distance / self.args.radius

        payoffs = node_obs[..., 5] 
    
        payoff_i = payoffs[b, src] 
        payoff_j = payoffs[b, dst]  
        
        payoff_difference = payoff_j - payoff_i

        normalized_payoff_diff = torch.tanh(payoff_difference).unsqueeze(-1)

        # d. Concatenate all edge features
        edge_attr = torch.cat([rel_pos, strat_one_hot, distance_normalized, normalized_payoff_diff], dim=-1) # (Num_Edges, 2+4+1=7)
        
        # Check if the final dimension matches self.edge_dim
        if edge_attr.shape[1] != self.edge_dim:
            raise ValueError(f"Constructed edge_attr dim {edge_attr.shape[1]} does not match expected edge_dim {self.edge_dim}")

        return edge_index, edge_attr


    # def forward(self, 
    #             node_obs: Tensor, # 形状: (M, N, node_feat_dim)
    #             adj: Tensor       # 形状: (M, N, N)
    #         ) -> Tensor:
    #     """
    #     Return: gnn_output
    #     - 如果 self.graph_aggr == "node":
    #         所有 M 个图中所有 N 个节点的嵌入, 形状 (M, N, self.out_dim)。
    #     - 如果 self.graph_aggr == "global":
    #         所有 M 个图的全局嵌入, 形状 (M, self.out_dim)。
    #     """
    #     M = node_obs.shape[0] # 图个数
    #     N = node_obs.shape[1] # 智能体个数 
    #     node_feat_dim = node_obs.shape[2] # 节点特征维度 

    #     # 构建 PyG Data 对象
    #     # print(f"[DEBUG]  adj: {adj.shape}, adj: {adj.dtype}")
    #     edge_index, edge_attr = TransformerConvNet.process_adj(adj, self.args.max_edge_dist)  # 处理 M 个距离邻接矩阵              
    #     x_nodes = node_obs.reshape(-1, node_feat_dim) # 形状 (M * N, node_feat_dim)    
    #     batch_index = torch.arange(M, device=x_nodes.device).repeat_interleave(N) # 形状 (M * N)，值从 0 到 M-1     
    #     pyg_batch_data = Data(x=x_nodes, 
    #                         edge_index=edge_index, 
    #                         edge_attr=edge_attr, 
    #                         batch=batch_index, # 用于将扁平化的 x_nodes 中的节点正确地分组回它们各自的原始图
    #                         )

    #     # 送入核心 GNN 网络 (self.gnn_core 即 TransformerConvNet)
    #     gnn_output = self.gnn_core(pyg_batch_data) 

    #     # 对结果进行最终的形状进行检查和调整
    #     if self.graph_aggr == 'node':
    #         # 形状应该是 (M * N, self.out_dim)
    #         if gnn_output.shape[0] != M * N: # 形状检查
    #             raise ValueError(f"GNN output shape {gnn_output.shape} unexpected for node-level aggregation. "
    #                              f"Expected first dimension to be M*N = {M * N}.")
    #         return gnn_output.view(M, N, self.out_dim)   # reshape 成 (M, N, self.out_dim)      
    #     elif self.graph_aggr == 'global':
    #         # 形状应该直接是 (M, self.out_dim)
    #         if gnn_output.shape[0] != M: # 形状检查
    #             raise ValueError(f"GNN output shape {gnn_output.shape} unexpected for global-level aggregation. "
    #                             f"Expected first dimension to be M = {M}.")
    #         if gnn_output.dim() != 2 or gnn_output.shape[1] != self.out_dim : # 形状检查
    #             raise ValueError(f"GNN output shape {gnn_output.shape} unexpected for global-level aggregation. "
    #                             f"Expected shape (M, out_dim) = ({M}, {self.out_dim}).")
    #         return gnn_output
        
    def forward(self, 
                node_obs: Tensor, # 形状: (M, N, node_feat_dim)
                adj: Tensor,       # 形状: (M, N, N)
                # full_node_obs_for_edge: Optional[Tensor] = None
            ) -> Tensor:
        M, N, _ = node_obs.shape

        if self.use_mini_node_feat:
            # For Actor's GNN: slice out the strategy feature (index 4)
            gnn_node_input = node_obs[..., 4:5]
        else:
            # For Critic's GNN: use the full features
            gnn_node_input = node_obs
        
        edge_index, edge_attr = self._process_graph_data(node_obs, adj)

        # # [MODIFIED] Call the new instance method to process graph data
        # edge_creation_obs = full_node_obs_for_edge if full_node_obs_for_edge is not None else node_obs
        # edge_index, edge_attr = self._process_graph_data(edge_creation_obs, adj)
        
        x_nodes = gnn_node_input.reshape(-1, gnn_node_input.shape[-1])
        batch_index = torch.arange(M, device=x_nodes.device).repeat_interleave(N)
        
        pyg_batch_data = Data(x = x_nodes, 
                            edge_index = edge_index, 
                            edge_attr = edge_attr,
                            batch = batch_index)

        gnn_output = self.gnn_core(pyg_batch_data) 

        # ... (The rest of the forward pass logic for reshaping the output is correct and remains unchanged) ...
        if self.graph_aggr == 'node':
            return gnn_output.view(M, N, self.out_dim)      
        elif self.graph_aggr == 'global':
            return gnn_output 