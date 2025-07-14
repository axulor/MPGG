import gym
import argparse
import torch
from torch import Tensor
from typing import Tuple, Any # Any 用于 args
from algorithms.graph_actor_critic import GR_Actor, GR_Critic # 假设这两个类后续会被修改
from algorithms.utils.gnn import GNNBase # 导入 GNNBase
from utils.util import update_linear_schedule, get_shape_from_obs_space, check # 导入 get_shape_from_obs_space

class GR_MAPPOPolicy:
    """
    MAPPO Policy 类。封装 Actor 和 Critic 网络，以及 GNN 特征提取器
    负责计算动作和价值函数预测
    """

    def __init__(
        self,
        args: argparse.Namespace, 
        obs_space: gym.Space,
        node_obs_space: gym.Space, 
        edge_obs_space: gym.Space, 
        action_space: gym.Space,
        device=torch.device("cpu"),
    ) -> None:
        
        # 接收参数
        self.device = device
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = args.num_agents 

        # 原始观测空间信息 
        self.obs_space = obs_space
        self.node_obs_space = node_obs_space 
        self.edge_obs_space = edge_obs_space
        self.action_space = action_space

        # GNN 需要的维度信息
        full_node_feat_dim = get_shape_from_obs_space(node_obs_space)[-1] # This is 6
        edge_feat_dim = get_shape_from_obs_space(edge_obs_space)[-1]
        mini_node_feat_dim = 1 # We use only strategy for Actor's GNN

        # GNN 特征提取器
        self.actor_gnn = GNNBase( # Actor 使用的 GNN，输出节点级嵌入
            args,
            node_obs_dim = mini_node_feat_dim,
            edge_dim = edge_feat_dim,
            graph_aggr = "node",
            use_mini_node_feat=True,
            device = self.device
        )
        actor_gnn_output_dim = self.actor_gnn.out_dim # 单个节点嵌入的维度

        self.critic_gnn = GNNBase( # Critic 使用的 GNN，输出图级全局嵌入
            args,
            node_obs_dim = full_node_feat_dim,
            edge_dim = edge_feat_dim,
            graph_aggr = "global",
            use_mini_node_feat=False,
            device = self.device
        )
        critic_gnn_output_dim = self.critic_gnn.out_dim # 全局图嵌入的维度

        #  Actor 和 Critic 网络
        actor_mlp_input_dim = actor_gnn_output_dim
        self.actor = GR_Actor( # 
            args,
            actor_mlp_input_dim, # 输入维度
            action_space,
            device = self.device
        )

        node_obs_processor_dim = args.hidden_size // 2
        critic_mlp_input_dim = node_obs_processor_dim + critic_gnn_output_dim
        self.critic = GR_Critic( 
            args,
            node_obs_space,  
            critic_mlp_input_dim, # 输入维度
            device = self.device
        )
        
        # 初始化优化器
        actor_params = list(self.actor.parameters()) + list(self.actor_gnn.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        
        critic_params = list(self.critic.parameters()) + list(self.critic_gnn.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode: int, episodes: int) -> None:
        """
        学习率衰减        
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, 
                    obs: Tensor,                # (M*N, D_obs)
                    node_obs: Tensor,           # (M, N, D_obs)
                    adj: Tensor,                # (M, N, N)
                    agent_id: Tensor,           # (M*N, 1) 
                    env_id: Tensor,             # (M*N, 1)
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return:
        - actions: 样本中每个智能体采取的动作
        - action_log_probs: 动作的对数概率
        """

        # # Actor GNN 特征提取
        # actor_gnn_output = self.actor_gnn(node_obs, adj)  # (M, N, D_actor_gnn_out)
        # env_ids = env_id.squeeze(-1).long() # 形状 (M*N)
        # agent_ids = agent_id.squeeze(-1).long() # 形状 (M*N)
        # actor_gnn_feat = actor_gnn_output[env_ids, agent_ids, :] # 取出对应环境, 对应智能体的嵌入, 形状 (M*N, D_actor_gnn_out)

        # actor_gnn_input = node_obs[..., 4:5] # Extract just the strategy feature
        actor_gnn_output = self.actor_gnn(
            node_obs, # Pass 1D features
            adj,
            # full_node_obs_for_edge=node_obs # Pass 6D features for edge calculation
        )
        env_ids = env_id.squeeze(-1).long()
        agent_ids = agent_id.squeeze(-1).long()
        actor_gnn_feat = actor_gnn_output[env_ids, agent_ids]

        # 送入 Actor 网络前向传播获取动作和动作概率 
        actions, action_log_probs = self.actor.forward(actor_gnn_feat) # 拼接各自的观测
        return actions, action_log_probs
    
    def get_values(self, 
                node_obs: Tensor,    # (M, N, D_obs)
                adj: Tensor,         # (M, N, N)
                ) -> Tensor:
        """
        Return:
        - values: 全局观测下的状态价值
        """
        # # Critic GNN 特征提取
        # critic_gnn_feat = self.critic_gnn(node_obs, adj) # 形状 (M, D_critic_gnn_output)
        critic_gnn_feat = self.critic_gnn(node_obs, adj)
        # 调用 Critic 网络前向传播获取价值
        values = self.critic.forward(node_obs, critic_gnn_feat) # 拼接全局观测

        # print(f"  values: {values.shape}, dtype: {values.dtype}, device: {values.device}")
        
        return values


    def evaluate_actions(self, obs, node_obs, adj, agent_id, env_id, actions):
        # --- Actor Path ---
        actor_gnn_output = self.actor_gnn(node_obs, adj)
        
        batch_indices = torch.arange(node_obs.shape[0], device=self.device)
        agent_ids = agent_id.squeeze(-1).long()
        actor_gnn_feat = actor_gnn_output[batch_indices, agent_ids]
        
        action_log_probs, dist_entropy = self.actor.evaluate_actions(actor_gnn_feat, actions)

        # --- Critic Path ---
        critic_gnn_feat = self.critic_gnn(node_obs, adj)
        values = self.critic.forward(node_obs, critic_gnn_feat)
        
        return values, action_log_probs, dist_entropy


    def prep_training(self) -> None:
        """
        Sets all networks to training mode.
        """
        self.actor.train()
        self.critic.train()
        self.actor_gnn.train()
        self.critic_gnn.train()

    def prep_evaluating(self) -> None:
        """
        Sets all networks to evaluation mode.
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_gnn.eval()
        self.critic_gnn.eval()

    # --- END OF NEW METHODS ---
    
