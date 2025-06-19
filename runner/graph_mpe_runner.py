import time
import numpy as np
from numpy import ndarray as arr # 类型别名，方便书写
from typing import Tuple, Dict, List 
import torch
import os

# from torch.utils.tensorboard import SummaryWriter # 可以用这个或 tensorboardX
from tensorboardX import SummaryWriter # 使用 tensorboardX
from algorithms.graph_mappo import GR_MAPPO 
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
from utils.graph_buffer import GraphReplayBuffer #  导入 Buffer 类
from utils.fast_eval import FastEvaluate  # 导入策略评估类


def _t2n(x):
    """将 PyTorch Tensor 转换为 NumPy Array"""
    return x.detach().cpu().numpy()

class GMPERunner: 
    """
    执行 MPGG (图 MPE) 环境的训练、评估和数据收集
    """

    def __init__(self, config: Dict):
        """
        Args:
            config (Dict)
        """
        # Runner 初始化
        self.all_args = config["all_args"]      # 存储所有超参数的配置对象, types.SimpleNamespace
        self.envs = config["envs"]              # 训练环境, envs.env_wrappers.GraphDummyVecEnv 
        self.eval_envs = config["eval_envs"]    # 评估环境, envs.env_wrappers.GraphDummyVecEnv
        self.device = config["device"]          # 计算设备, torch.device 

        # 训练设置参数
        self.num_agents = self.all_args.num_agents                          # 环境中由策略控制的智能体数量 int
        self.num_env_steps = self.all_args.num_env_steps                    # 整个训练过程将要运行的环境交互总次数 int
        self.episode_length = self.all_args.episode_length                  # Replay Buffer 中存储的时序轨迹的长度 int 
        self.n_rollout_threads = self.all_args.n_rollout_threads            # 用于数据搜集的并行环境实例数量 int
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 用于策略评估的并行环境实例数量 int
        self.hidden_size = self.all_args.hidden_size                        # 神经网络隐藏层的大小/维度 int


        # 时间间隔参数
        self.save_interval = self.all_args.save_interval    # 每隔 save_interval 个回合，保存当前的 Actor 和 Critic 网络权重
        self.log_interval = self.all_args.log_interval      # 每隔 log_interval 个回合, 写入训练日志
        self.use_eval = self.all_args.use_eval              # 训练暂停，使用当前的策略在独立的评估环境中运行，并记录性能指标     
        self.eval_interval = self.all_args.eval_interval    # 每隔 eval_interval 个回合, 启动评估
        self.global_reset_interval = self.all_args.global_reset_interval # 每隔 global_reset_interval 个回合, 重置所有并行环境


        # 目录参数
        self.model_dir = self.all_args.model_dir        # 预训练模型目录        
        self.run_dir = config["run_dir"]                # 运行结果总目录        
        self.log_dir = str(self.run_dir / "logs")       # TensorBoard 日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)      # 初始化 TensorBoard writer
        self.save_dir = str(self.run_dir / "models")    # 模型保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        # 初始化策略、训练器和 Buffer

        # 确定 Critic 输入的中心化观测空间
        share_obs_space = self.envs.share_observation_space[0] 

        # 初始化策略网络 (Actor 和 Critic)
        print("  (Runner) 初始化策略网络...")

        # 图环境 Policy 初始化 (保持上次修正后的参数列表)
        self.policy = GR_MAPPOPolicy(
            self.all_args,                      # 1. args
            self.envs.observation_space[0],     # 2. 单个智能体的个体局部观测空间
            share_obs_space,                    # 3. 传入 Critic的全局观测空间
            self.envs.node_observation_space[0],# 4. 传入 GNN的图节点特征空间
            self.envs.edge_observation_space[0],# 5. 传入 GNN的图边特征空间
            self.envs.action_space[0],          # 6. 单个智能体的动作空间
            device=self.device                  # 7. device
        )

        # 如果指定了预训练模型目录，则加载模型参数
        if self.model_dir is not None:
            print(f"  (Runner) 从目录加载预训练模型: {self.model_dir}")
            self.restore() # 调用加载模型的函数

        # 初始化训练器 GR_MAPPO
        print("  (Runner) 初始化训练器...")
        self.trainer = GR_MAPPO(self.all_args, self.policy, device=self.device)
        print("  (Runner) 训练器初始化完成.")

        # 初始化经验回放缓冲区
        print("  (Runner) 初始化 Replay Buffer...")
        self.buffer = GraphReplayBuffer(
            self.all_args,
            self.envs.observation_space[0],
            share_obs_space,
            self.envs.node_observation_space[0],
            self.envs.agent_id_observation_space[0],
            self.envs.share_agent_id_observation_space[0],
            self.envs.adj_observation_space[0], 
            self.envs.action_space[0],
        )

        # 实例化评估器
        self.evaluator = None
        if self.all_args.use_eval and self.eval_envs is not None:
            print("  (Runner) 初始化评估器...")
            try:
                self.evaluator = FastEvaluate(
                    all_args=self.all_args,     # 传递完整的配置对象
                    policy=self.policy,         # 传递 self.policy
                    eval_envs=self.eval_envs,   # 传递评估环境
                    run_dir=self.run_dir,       # 将 Runner 自己的 run_dir 传递过去
                )
                print("  (Runner) 评估器初始化完成.")
            except Exception as e:
                print(f"警告：初始化评估器失败: {e}")
                self.evaluator = None
        else:
            self.evaluator = None

    def run(self):
        """主训练循环"""
        print("  (Runner) 开始 Warmup...")
        self.warmup() # 初始化 Buffer 中的第一个时间步数据
        reset_count = 0 # 全局重置计数器 
        print("  (Runner) Warmup 完成 ")

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print(f"  (Runner) 总训练步数: {self.num_env_steps}, 每回合步数: {self.episode_length}, 并行环境数: {self.n_rollout_threads}, 单个环境智能体数: { self.num_agents}")
        print(f"  (Runner) 将运行 {episodes} 个学习更新周期 ")

        # 核心训练循环
        for episode in range(episodes):
            # 学习率线性下降
            self.trainer.policy.lr_decay(episode, episodes)
            
            # 环境定期重置
            if reset_count >= self.global_reset_interval:
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 达到全局重置间隔。正在重置所有环境...")
                self.warmup() # 调用 warmup 来执行重置和 Buffer[0] 的初始化
                reset_count = 0 # 重置计数器
                print(f"  (Runner) 所有环境已重置并 Warmup 完成。")

            if episode == 0:
                print(f"  执行初始评估...")
                self.fast_eval(episode)
                print("  (Runner) 评估完成.")

            segment_coop_rates = np.full((self.episode_length, self.n_rollout_threads), np.nan, dtype=np.float32) # 存储合作率数据
            segment_avg_rewards = np.full((self.episode_length, self.n_rollout_threads), np.nan, dtype=np.float32) # 存储奖励数据

            # 并行 rollout 循环
            for step in range(self.episode_length):                            
                values, actions, action_log_probs = self.sample_policy_outputs()    # 采样动作         
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions)    # 执行动作
                # 处理环境指标信息
                for thread in range(self.n_rollout_threads):
                    if infos[thread] and len(infos[thread]) > 0:
                        global_info_for_thread_step = infos[thread][0]                         
                        coop_rate = global_info_for_thread_step.get("step_cooperation_rate")
                        avg_reward = global_info_for_thread_step.get("step_avg_reward")
                        if coop_rate is not None:
                            segment_coop_rates[step, thread] = coop_rate
                        if avg_reward is not None:
                            segment_avg_rewards[step, thread] = avg_reward
                # 处理中心化观测数据
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
                share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
                share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1)

                # print(f"[DEBUG]  share_obs: {share_obs.shape}, share_obs: {share_obs.dtype}")
                # print(f"[DEBUG]  obs: {obs.shape}, obs: {obs.dtype}")
                # print(f"[DEBUG]  node_obs: {node_obs.shape}, node_obs: {node_obs.dtype}")
                # print(f"[DEBUG]  adj: {adj.shape}, adj: {adj.dtype}")
                # print(f"[DEBUG]  agent_id: {agent_id.shape}, agent_id: {agent_id.dtype}")
                # print(f"[DEBUG]  share_agent_id: {share_agent_id.shape}, share_agent_id: {share_agent_id.dtype}")
                # print(f"[DEBUG]  actions: {actions.shape}, actions: {actions.dtype}")
                # print(f"[DEBUG]  action_log_probs: {action_log_probs.shape}, action_log_probs: {action_log_probs.dtype}")
                # print(f"[DEBUG]  values: {values.shape}, values: {values.dtype}")
                # print(f"[DEBUG]  rewards: {rewards.shape}, rewards: {rewards.dtype}")
                # print(f"[DEBUG]  dones: {dones.shape}, dones: {dones.dtype}")

                # 向 buffer 中插入数据 
                self.buffer.insert(
                    share_obs,
                    obs,
                    node_obs,
                    adj,
                    agent_id,
                    share_agent_id,
                    actions,              
                    action_log_probs,     
                    values,               
                    rewards,              
                    dones,
                )

            # rollout 结束后为 buffer 计算 returns
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 计算 Return...")
            with torch.no_grad():

                self.trainer.prep_evaluating() # 置为评估模式

                # 对 rollout segment 中的最后一个状态 s_L 进行价值估计
                buffer_idx_L = self.buffer.episode_length
                node_obs = self.buffer.node_obs[buffer_idx_L][:, 0, :, :] # 所有智能体都是相同的
                adj = self.buffer.adj[buffer_idx_L][:, 0, :, :]  # 所有智能体都是相同的
                share_obs = np.concatenate(self.buffer.share_obs[buffer_idx_L])

                # 将 NumPy 数组转换为 PyTorch Tensors 并移到设备
                share_obs = torch.from_numpy(share_obs).float().to(self.device)
                node_obs = torch.from_numpy(node_obs).float().to(self.device) 
                adj = torch.from_numpy(adj).float().to(self.device)
                obs = torch.from_numpy(obs).float().to(self.device) # 特征数据是 float32

                # 打印价值估计的准备参数
                print(f"--- Printing parameters for estimating ---")
                print(f"  node_obs: {node_obs.shape}, dtype: {node_obs.dtype}, device: {node_obs.device}")
                print(f"  adj: {adj.shape}, dtype: {adj.dtype}, device: {adj.device}")
                print(f"  share_obs: {share_obs.shape}, dtype: {share_obs.dtype}, device: {share_obs.device}")
                print(f"--- Printing completed ---")

                next_values = self.trainer.policy.get_values(
                    node_obs,
                    adj,
                    share_obs,                  
                )
                next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))            
            self.buffer.compute_returns(next_values, self.trainer.value_normalizer) # 在 buffer 中计算 returns 
            print("  (Runner) 计算完成.")

            # 从 buffer 中采样训练网络
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 开始训练网络...")
            self.trainer.prep_training() # 设置为训练模式
            train_infos = self.trainer.train(self.buffer) # 网络训练循环
            self.buffer.after_update() # 训练结束处理buffer
            print(" (Runner) 训练完成.")
        
            # 处理 infos
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads 
            print(f"--- {episode + 1}/{episodes} 结束 (总步数: {total_num_steps}) ---")
            # 训练指标
            if train_infos:
                print("  训练指标 (Train Infos):")
                for key, value in train_infos.items():
                    if isinstance(value, (float, int, np.number)):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        print(f"    {key}: {value.item():.4f}")
            else:
                print("  训练指标 (Train Infos): 无")

            # 环境指标
            env_infos = self.process_infos(segment_coop_rates, segment_avg_rewards)            
            if env_infos:
                print("  环境指标 (Env Infos ):")
                for key, value in env_infos.items():
                    if isinstance(value, (float, int, np.number)):
                        print(f"    {key}: {value:.4f}")
            else:
                print("  环境指标 (Env Infos): 无")

            # 保存模型
            if (episode + 1) % self.save_interval == 0 or episode == episodes - 1:
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 保存模型...")
                self.save()
                print("  (Runner) 模型已保存.")

            # 保存日志
            if (episode + 1) % self.log_interval == 0:
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 记录日志...")
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                print("  (Runner) 日志记录完成.")

            # 评估模型                
            if self.use_eval and ((episode + 1) % self.eval_interval == 0 or episode == episodes - 1):
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 执行评估...")
                self.fast_eval(episode)
                print("  (Runner) 评估完成.")

            reset_count +=1 

        print("所有训练回合已完成. ")


    def warmup(self):
        """
        初始化 Replay Buffer 中的第一个时间步的数据 (s_0).
        This is called once at the very beginning of training.
        """
        obs, agent_id, node_obs, adj = self.envs.reset() # This resets all parallel envs
        
        # 中心化观测
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
        share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1)

        # Store initial observations at buffer.obs[0], etc.
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()
        
        self.buffer.dones[0] = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1),
            dtype=np.float32
        )

        self.buffer.step = 0 # Explicitly set buffer's circular pointer to 0
    

    @torch.no_grad() 
    def sample_policy_outputs(self) -> Tuple[arr, arr, arr]: 
        """
        根据 Buffer 中的当前状态，调用策略网络获取动作、动作对数概率和价值估计。

        Return: 
        - values: 动作价值
        - actions: 动作
        - action_log_probs: 动作概率 
        """
        self.trainer.prep_evaluating() # 将 Actor 和 Critic 设置为评估模式

        # 从 Buffer 获取当前时间步 k = self.buffer.step 的数据
        buffer_step = self.buffer.step
        obs = np.concatenate(self.buffer.obs[buffer_step]) # Actor 和 Critic 的 MLP 输入所需的扁平化观测
        share_obs = np.concatenate(self.buffer.share_obs[buffer_step])
        node_obs = self.buffer.node_obs[buffer_step][:, 0, :, :]  # GNN 输入的 M 个独特的全局图数据
        adj = self.buffer.adj[buffer_step][:, 0, :, :]
        agent_id = np.concatenate(self.buffer.agent_id[buffer_step]) # 用于 Actor GNN 结果挑选的 ID
        env_indices = np.arange(self.n_rollout_threads) 
        env_id = np.repeat(env_indices, self.num_agents).reshape(-1, 1)

        # # 打印 NumPy 形状和类型
        # print(f"--- Print the shape and type of NumPy in sample_policy_outputs ---") 
        # print(f"  Numpy obs_np shape: {obs.shape}, dtype: {obs.dtype}")
        # print(f"  Numpy share_obs_np shape: {share_obs.shape}, dtype: {share_obs.dtype}")
        # print(f"  Numpy node_obs_np shape: {node_obs.shape}, dtype: {node_obs.dtype}")
        # print(f"  Numpy adj_np shape: {adj.shape}, dtype: {adj.dtype}")
        # print(f"  Numpy agent_id_np shape: {agent_id.shape}, dtype: {agent_id.dtype}")
        # print(f"  Numpy env_id_np shape: {env_id.shape}, dtype: {env_id.dtype}")
        # print(f"--- Printing completed ---")

        # 将 NumPy 数组转换为 PyTorch Tensors 并移到设备
        obs = torch.from_numpy(obs).float().to(self.device) # 特征数据是 float32
        share_obs = torch.from_numpy(share_obs).float().to(self.device)
        node_obs = torch.from_numpy(node_obs).float().to(self.device) 
        adj = torch.from_numpy(adj).float().to(self.device)
        agent_id = torch.from_numpy(agent_id).long().to(self.device) # ID 数据是 long (int64)
        env_id = torch.from_numpy(env_id).long().to(self.device) 
        
        # # 打印tensor的形状和数据
        # print(f"  Torch obs_tensor: {obs.shape}, dtype: {obs.dtype}, device: {obs.device}")
        # print(f"  Torch share_obs_tensor: {share_obs.shape}, dtype: {share_obs.dtype}, device: {share_obs.device}")
        # print(f"  Torch node_obs_tensor: {node_obs.shape}, dtype: {node_obs.dtype}, device: {node_obs.device}")
        # print(f"  Torch adj_tensor: {adj.shape}, dtype: {adj.dtype}, device: {adj.device}")
        # print(f"  Torch agent_id_tensor: {agent_id.shape}, dtype: {agent_id.dtype}, device: {agent_id.device}")
        # print(f"  Torch env_id_tensor: {env_id.shape}, dtype: {env_id.dtype}, device: {env_id.device}")
        # print(f"--- Printing completed ---")

        # 调用策略网络获取动作和动作对数概率
        torch_actions, torch_action_log_probs = self.trainer.policy.get_actions(
            obs,        # (M*N, D_obs)
            node_obs,   # (M, N_nodes, D_node_raw)
            adj,        # (M, N_nodes, N_nodes)
            agent_id,   # (M*N, 1)
            env_id      # (M*N, 1)
        )
        
        # 调用策略网络获取价值估计
        torch_values = self.trainer.policy.get_values(
            node_obs,       # (M, N_nodes, D_node_raw)
            adj,            # (M, N_nodes, N_nodes)
            share_obs  
        )

        # 将结果从 Tensor 转换为 NumPy 并恢复线程维度
        values = np.array(np.split(_t2n(torch_values), self.n_rollout_threads)) 
        actions = np.array(np.split(_t2n(torch_actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(torch_action_log_probs), self.n_rollout_threads))

        # print(f"--- Printing sample_policy_outputs shape ---")
        # print(f"  values: {values.shape}, dtype: {values.dtype}, device: {values.device}")
        # print(f"  actions: {actions.shape}, dtype: {actions.dtype}, device: {actions.device}")
        # print(f"  action_log_probs: {action_log_probs.shape}, dtype: {action_log_probs.dtype}, device: {action_log_probs.device}")
        # print(f"--- Printing completed ---")

        return values, actions, action_log_probs


    @torch.no_grad()
    def fast_eval(self, episode: int): 
        """
        执行快速策略评估
        """
        print(f"DEBUG: GMPERunner.fast_eval called for episode {episode}. Evaluator ID: {id(self.evaluator) if self.evaluator else 'None'}") # <--- 新增打印
        print(f"--- 开始评估 (在网络更新次数为 {episode} 时) ---")
        # 执行评估并获取结果
        eval_results = self.evaluator.eval_policy() # 返回评估结果

        # 绘制并保存结果图
        if eval_results:
            self.evaluator.plot_results(eval_results, episode)
        else:
            print("评估未产生数据。")
        print(f"--- 评估结束 (在网络更新次数为 {episode} 时) ---")

    def save(self):
        """保存策略的 Actor 和 Critic 网络参数 """
        # 检查保存目录是否存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"  创建模型保存目录: {self.save_dir}")

        actor_save_path = str(self.save_dir) + "/actor.pt"
        critic_save_path = str(self.save_dir) + "/critic.pt"
        print(f"  保存 Actor 到: {actor_save_path}")
        print(f"  保存 Critic 到: {critic_save_path}")
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), actor_save_path)
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), critic_save_path)

    def restore(self):
        """从指定目录加载预训练模型参数 """
        if self.model_dir is None:
            print("错误：未指定模型目录 (model_dir is None)，无法加载。")
            return
        actor_load_path = str(self.model_dir) + "/actor.pt"
        critic_load_path = str(self.model_dir) + "/critic.pt"

        if not os.path.exists(actor_load_path):
            print(f"错误：找不到 Actor 模型文件: {actor_load_path}")
            return
        print(f"  从文件加载 Actor: {actor_load_path}")
        policy_actor_state_dict = torch.load(actor_load_path, map_location=self.device)
        self.policy.actor.load_state_dict(policy_actor_state_dict)

        if not os.path.exists(critic_load_path):
            print(f"错误：找不到 Critic 模型文件: {critic_load_path}")
            return
        print(f"  从文件加载 Critic: {critic_load_path}")
        policy_critic_state_dict = torch.load(critic_load_path, map_location=self.device)
        self.policy.critic.load_state_dict(policy_critic_state_dict)


    def process_infos(self, 
                    segment_coop_rates_data: np.ndarray, 
                    segment_avg_rewards_data: np.ndarray
                    ) -> Dict[str, float]: 
        """
        Return:
        - env_infos_to_log: 聚合后标量指标的字典
        """
        env_infos_to_log = {} 
        
        # 计算整个 segment (所有 step, 所有 thread) 的平均合作率
        if segment_coop_rates_data.size > 0: # 确保数组不为空
            mean_coop_rate_for_segment = np.nanmean(segment_coop_rates_data)
            env_infos_to_log["cooperation_rate"] = mean_coop_rate_for_segment
        else:
            env_infos_to_log["cooperation_rate"] = np.nan

        # 计算整个 segment (所有 step, 所有 thread) 的平均奖励
        if segment_avg_rewards_data.size > 0: # 确保数组不为空
            mean_avg_reward_for_segment = np.nanmean(segment_avg_rewards_data)
            env_infos_to_log["avg_reward"] = mean_avg_reward_for_segment
        else:
            env_infos_to_log["avg_reward"] = np.nan

        return env_infos_to_log


    def log_train(self, train_infos: Dict, total_num_steps: int):
        """ 记录训练信息到 TensorBoard """
        for key, value in train_infos.items():
            scalar_to_log = None # 用于存储最终要记录的标量值

            if isinstance(value, (int, float, np.number)):
                scalar_to_log = float(value) # 统一转换为 float
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar_to_log = value.item() # 从单元素 Tensor 中取出 Python 数值

            self.writter.add_scalar(f"train/{key}", scalar_to_log, total_num_steps)


    def log_env(self, env_infos: Dict, total_num_steps: int):
        """ 记录环境信息到 TensorBoard """
        for key, value in env_infos.items():
            if isinstance(value, (int, float, np.number)):
                self.writter.add_scalar(f"env/{key}", value, total_num_steps)