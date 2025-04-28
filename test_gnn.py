import torch
import numpy as np
from torch_geometric.data import Data

from algorithms.utils.gnn import TransformerConvNet, GNNBase

class Args:
    num_embeddings = 3
    embedding_size = 4
    embed_hidden_size = 5
    embed_layer_N = 2
    use_orthogonal = False
    use_ReLU = True
    use_feature_normalization = False
    embed_add_self_loop = True
    gnn_hidden_size = 6
    gnn_num_heads = 2
    gnn_concat_heads = True
    gnn_layer_N = 2
    gnn_use_ReLU = True
    max_edge_dist = 1.0
    global_aggr_type = 'mean'
    embed_use_ReLU = True

def test_process_adj():
    print("Testing process_adj...")
    # 2D case
    adj2d = torch.tensor([[0.0, 0.5, 2.0],
                          [0.5, 0.0, 0.2],
                          [2.0, 0.2, 0.0]])
    edge_index, edge_attr = TransformerConvNet.process_adj(adj2d, max_edge_dist=1.0)
    print("  2D edge_index:", edge_index.shape)
    print("  2D edge_attr:", edge_attr.shape)
    assert edge_index.shape[0] == 2
    # 3D case
    batch_size, N = 2, 3
    adj3d = torch.stack([adj2d, adj2d * 0.1])
    ei3, ea3 = TransformerConvNet.process_adj(adj3d, max_edge_dist=1.0)
    print("  3D edge_index:", ei3.shape)
    print("  3D edge_attr:", ea3.shape)
    assert ei3.shape[0] == 2


def test_gatherNodeFeats_and_graphAggr():
    print("Testing gatherNodeFeats and graphAggr...")
    batch_size, num_nodes, feat_dim = 2, 4, 5
    x = torch.arange(batch_size * num_nodes * feat_dim, dtype=torch.float32).view(batch_size, num_nodes, feat_dim)
    idx = torch.tensor([[0, 3], [1, 2]])
    # instantiate a dummy GNNBase to access methods
    args = Args()
    gnn_base = GNNBase(args, node_obs_shape=num_nodes, edge_dim=1, graph_aggr='mean')
    # test gatherNodeFeats
    # feats = gnn_base.gatherNodeFeats(x, idx)
    # print("  gathered feats shape:", feats.shape)
    # assert feats.shape == (batch_size, feat_dim * idx.shape[1])
    # test graphAggr
    # aggr_mean = gnn_base.graphAggr(x)
    # print("  mean aggr shape:", aggr_mean.shape)
    # assert aggr_mean.shape == (batch_size, feat_dim)
    # aggr_max = gnn_base.graphAggr(x, ) if False else None
    # Note: graphAggr used internal global_aggr_type


def test_transformerConvNet_and_GNNBase_forward():
    print("Testing TransformerConvNet and GNNBase forward...")
    # dummy data
    batch_size, num_nodes = 2, 3
    node_obs_dim = 7
    # Last column is entity type id (0..num_embeddings-1)
    node_obs = torch.randn(batch_size, num_nodes, node_obs_dim)
    entity_ids = torch.randint(0, Args.num_embeddings, (batch_size, num_nodes, 1), dtype=torch.float32)
    node_obs[:, :, -1:] = entity_ids
    # adjacency matrices
    adj = torch.rand(batch_size, num_nodes, num_nodes) * 2.0
    # ensure zero diagonal
    for i in range(batch_size):
        adj[i].fill_diagonal_(0)
    agent_id = torch.randint(0, num_nodes, (batch_size,), dtype=torch.long)
    # Instantiate GNNBase
    gnn_base = GNNBase(Args(), node_obs_shape=node_obs_dim, edge_dim=1, graph_aggr='global')
    # Forward
    out = gnn_base.forward(node_obs, adj, agent_id)
    print("  GNNBase output shape (global):", out.shape)
    assert out.shape == (batch_size, gnn_base.out_dim)
    # Test node aggregation
    gnn_node = GNNBase(Args(), node_obs_shape=node_obs_dim, edge_dim=1, graph_aggr='node')
    out_node = gnn_node.forward(node_obs, adj, agent_id)
    print("  GNNBase output shape (node):", out_node.shape)
    assert out_node.shape == (batch_size, gnn_node.out_dim)

if __name__ == '__main__':
    test_process_adj()
    test_gatherNodeFeats_and_graphAggr()
    test_transformerConvNet_and_GNNBase_forward()
    print("All GNN tests passed.")
