import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import math
import numpy as np
from gymnasium import spaces
from torch import Tensor


def init(module: nn.Module, weight_init, bias_init, gain: float = 1):
    """
    Initialize the weights and biases of a module.
    Args:
        module (nn.Module): The module to initialize.
        weight_init (callable): Function to initialize weights.
            init_method = nn.init.xavier_uniform_          # 默认选项，均匀分布初始化
            init_method = nn.init.orthogonal_              # 更适合 PPO/RNN，正交初始化
        bias_init (callable): Function to initialize biases.
            lambda x:nn.init.constant_(x, 0)     # 偏置初始化为 0
        gain (float, optional): Gain for the weight initialization. Default is 1.
    """
    weight_init(module.weight.data, gain=gain) # 获取权重数据并初始化
    bias_init(module.bias.data) # 获取偏置数据并初始化
    return module


def get_clones(module: nn.Module, N: int):
    """
    创建 N 个独立的、结构完全一样的 module 拷贝，放进一个 ModuleList 里，便于批量使用
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    """
    Check if the input is a numpy array or a PyTorch tensor and convert it to a PyTorch tensor if necessary.
    使用 torch.from_numpy 将 numpy.ndarray 转换为 torch.Tensor
    """
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def get_shape_from_obs_space(obs_space):
    """
    通用函数：从 observation space 中提取 shape 信息
    支持 Box、Sequence、Dict（特别是包含 self 和 neighbors 的结构）

    返回:
        tuple: 表示特征维度，例如 (4,), (2,), (1,)
    """
    if isinstance(obs_space, spaces.Box):
        return obs_space.shape

    elif isinstance(obs_space, spaces.Sequence):
        return get_shape_from_obs_space(obs_space.feature_space)

    elif isinstance(obs_space, spaces.Dict):
        keys = obs_space.spaces.keys()

        # 特殊处理图神经网络结构：包含 self 和 neighbors
        if "self" in keys and "neighbors" in keys:
            self_space = obs_space.spaces["self"]
            neighbor_space = obs_space.spaces["neighbors"].feature_space

            def _dim(subdict):
                dim = 0
                for v in subdict.spaces.values():
                    if isinstance(v, spaces.Discrete):
                        dim += 1
                    elif isinstance(v, spaces.Box):
                        dim += int(np.prod(v.shape))
                    else:
                        raise NotImplementedError(f"Unsupported subspace type: {type(v)}")
                return dim

            # GNN 需要：所有节点特征维度对齐
            max_dim = max(_dim(self_space), _dim(neighbor_space))
            return (max_dim,)  # ← 用于构建 GNNBase

        else:
            # 普通 Dict：统计全部维度总和
            dim = 0
            for v in obs_space.spaces.values():
                if isinstance(v, spaces.Discrete):
                    dim += 1
                elif isinstance(v, spaces.Box):
                    dim += int(np.prod(v.shape))
                else:
                    raise NotImplementedError(f"Unsupported Dict subspace type: {type(v)}")
            return (dim,)

    elif isinstance(obs_space, list):
        return tuple(obs_space)

    else:
        raise NotImplementedError(f"Unsupported obs_space type: {type(obs_space)}")



def obs_to_graph(obs: dict) -> Data:
    """
    将单个智能体的观测 dict 转换为 PyG 的图结构 Data。
    
    所有节点都以统一张量方式组织，无需使用 entity_type。

    节点特征：
        - 自身节点: [strategy, last_payoff]                    → shape: (2,)
        - 邻居节点: [angle, distance, strategy, last_payoff]    → shape: (4,)
    最终通过 padding 或拼接形成统一维度。

    边：
        - 从自身节点（节点编号 0）出发，连接到每个邻居
        - 边特征使用 distance 构成 (E, 1) 的张量

    Returns:
        torch_geometric.data.Data
    """

    # 构建自身节点特征（2维）
    self_strategy = float(obs["self"]["strategy"])
    self_payoff = float(obs["self"]["last_payoff"])
    self_feat = [self_strategy, self_payoff]

    # 所有节点特征列表
    x = [self_feat]  # 节点 0 是自身

    edge_index = [[], []]
    edge_attr = []

    for i, neighbor in enumerate(obs["neighbors"]):
        dx, dy = neighbor["relative_position"]
        distance = float(neighbor["distance"])
        angle = math.atan2(dy, dx)

        strategy = float(neighbor["strategy"])
        payoff = float(neighbor["last_payoff"])

        neighbor_feat = [angle, distance, strategy, payoff]
        x.append(neighbor_feat)

        # 建立边：从自身节点 0 → 邻居节点 i+1
        edge_index[0].append(0)
        edge_index[1].append(i + 1)
        edge_attr.append([distance])

    # 统一节点特征维度
    max_len = max(len(feat) for feat in x)
    for i in range(len(x)):
        pad_len = max_len - len(x[i])
        if pad_len > 0:
            x[i] += [0.0] * pad_len  # padding

    x = torch.tensor(x, dtype=torch.float32)                      # (num_nodes, feat_dim)
    edge_index = torch.tensor(edge_index, dtype=torch.long)      # (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)     # (num_edges, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def minibatch_generator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d) -> float:
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2

def get_grad_norm(it) -> float:
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def _t2n(x):
    return x.detach().cpu().numpy()
