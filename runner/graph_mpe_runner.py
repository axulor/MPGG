import time
import numpy as np
from numpy import ndarray as arr 
from typing import Tuple, Dict, List 
import torch
import os
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

        # 初始化策略网络 (Actor 和 Critic)
        print("  (Runner) 初始化策略网络...")

        # 图环境 Policy 初始化 (保持上次修正后的参数列表)
        self.policy = GR_MAPPOPolicy(
            self.all_args,                      # 1. args
            self.envs.observation_space[0],     # 2. 单个智能体的个体局部观测空间
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

        # [新增] 初始化起始episode
        self.start_episode = 0

        # 如果指定了预训练模型目录，则加载模型参数
        if self.model_dir is not None:
            print(f"  (Runner) Attempting to restore from directory: {self.model_dir}")
            # [修改] 我们现在加载最新的检查点
            checkpoint_path = os.path.join(self.model_dir, "checkpoint_latest.pt")
            resume_episode = self.restore(checkpoint_path)
            if resume_episode > 0:
                self.start_episode = resume_episode
            else:
                print("  (Runner) WARNING: Checkpoint restoration failed. Starting from scratch.")


        # 初始化经验回放缓冲区
        print("  (Runner) 初始化 Replay Buffer...")
        self.buffer = GraphReplayBuffer(
            self.all_args,
            self.envs.observation_space[0],
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
        for episode in range(self.start_episode, episodes):
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
                
                obs, rewards, adj, dones, infos = self.envs.step(actions)    # 执行动作,obs, agent_id, reward, adj, done, info   

                # 处理环境指标信息
                for thread in range(self.n_rollout_threads):
                    if infos[thread] and len(infos[thread]) > 0:
                        global_info_for_thread_step = infos[thread]                         
                        coop_rate = global_info_for_thread_step.get("step_cooperation_rate")
                        avg_reward = global_info_for_thread_step.get("step_avg_reward")
                        if coop_rate is not None:
                            segment_coop_rates[step, thread] = coop_rate
                        if avg_reward is not None:
                            segment_avg_rewards[step, thread] = avg_reward

                # 向 buffer 中插入数据 
                self.buffer.insert(obs, adj, actions, action_log_probs, values, rewards, dones)

            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 计算 Return...")
            with torch.no_grad():

                self.trainer.prep_evaluating() # 置为评估模式

                last_obs = torch.from_numpy(self.buffer.obs[-1]).float().to(self.device)
                last_adj = torch.from_numpy(self.buffer.adj[-1]).float().to(self.device)

                next_values = self.policy.get_values(last_obs, last_adj) # Returns (M, 1)
                next_values = _t2n(next_values).reshape(self.n_rollout_threads, 1, 1)
                next_values = np.repeat(next_values, self.num_agents, axis=1) #(M,N,1)

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
                self.save(episode)
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
        obs, adj = self.envs.reset() # resets all parallel envs

        self.buffer.obs[0] = obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.dones[0].fill(0.0)
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

        # Get data from buffer for the current step
        obs = torch.from_numpy(self.buffer.obs[self.buffer.step]).float().to(self.device)
        adj = torch.from_numpy(self.buffer.adj[self.buffer.step]).float().to(self.device)

        
        # Generate IDs on the fly
        agent_id = torch.arange(self.num_agents, device=self.device).unsqueeze(0).unsqueeze(-1).repeat(self.n_rollout_threads, 1, 1)
        env_id = torch.arange(self.n_rollout_threads, device=self.device).unsqueeze(1).unsqueeze(-1).repeat(1, self.num_agents, 1)

        # Reshape for policy's individual agent inputs
        obs_flatten = obs.view(-1, obs.shape[-1])
        agent_id_flatten = agent_id.view(-1, 1)
        env_id_flatten = env_id.view(-1, 1)

        # print(f"[DEBUG]in sample_policy_outputs,  obs_flatten: {obs_flatten.shape}, obs_flatten: {obs_flatten.dtype}") # (200,6)
        # print(f"[DEBUG]in sample_policy_outputs,  obs: {obs.shape}, obs: {obs.dtype}") # (8,25,6)
        # print(f"[DEBUG]in sample_policy_outputs,  adj: {adj.shape}, adj: {adj.dtype}") # (8,25,25)
        # print(f"[DEBUG]in sample_policy_outputs,  agent_id_flatten: {agent_id_flatten.shape}, agent_id_flatten: {agent_id_flatten.dtype}") # (200,1)
        # print(f"[DEBUG]in sample_policy_outputs,  env_id_flatten: {env_id_flatten.shape}, env_id_flatten: {env_id_flatten.dtype}") # (200,1)

        # Get actions from policy
        # Pass both individual (reshaped) and global (original) views of obs
        torch_actions, torch_action_log_probs = self.policy.get_actions(
            obs_flatten, obs, adj, agent_id_flatten, env_id_flatten
        )

        # 调用策略网络获取价值估计
        torch_values = self.policy.get_values(obs, adj) # Returns (M, 1)
        values_per_agent = torch_values.repeat_interleave(self.num_agents, dim=0) # (M*N, 1)

        # 将结果从 Tensor 转换为 NumPy 并恢复线程维度
        values = np.array(np.split(_t2n(values_per_agent), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(torch_actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(torch_action_log_probs), self.n_rollout_threads))

        # print(f"[DEBUG]in sample_policy_outputs,  values: {values.shape}, values: {values.dtype}") # (M,N,1)
        # print(f"[DEBUG]in sample_policy_outputs,  actions: {actions.shape}, actions: {actions.dtype}") # (M,N,2)
        # print(f"[DEBUG]in sample_policy_outputs,  action_log_probs: {action_log_probs.shape}, action_log_probs: {action_log_probs.dtype}") # (M,N,1)
        
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


    def save(self, episode: int):
        """
        保存一个包含完整训练状态的检查点。

        Args:
            episode (int): 当前的 episode 编号，用于命名文件。
        """
        # 确保保存目录存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 1. 创建一个包含所有可恢复状态的字典
        # 这是实现断点续训的关键
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.policy.actor.state_dict(),
            'critic_state_dict': self.policy.critic.state_dict(),
            'actor_optimizer_state_dict': self.policy.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.policy.critic_optimizer.state_dict(),
        }

        # 如果使用了 GNN，也保存 GNN 的状态
        if hasattr(self.policy, 'actor_gnn'):
            checkpoint['actor_gnn_state_dict'] = self.policy.actor_gnn.state_dict()
        if hasattr(self.policy, 'critic_gnn'):
            checkpoint['critic_gnn_state_dict'] = self.policy.critic_gnn.state_dict()
            
        # 如果使用了价值归一化 (PopArt 或 ValueNorm)，保存其状态
        if self.trainer.value_normalizer is not None and hasattr(self.trainer.value_normalizer, 'state_dict'):
            checkpoint['value_normalizer_state_dict'] = self.trainer.value_normalizer.state_dict()

        # 2. 保存一个以 episode 编号命名的、永久性的快照
        # episode 编号从0开始，我们保存时用 ep1, ep2... 更直观
        snapshot_path = os.path.join(self.save_dir, f"checkpoint_ep{episode + 1}.pt")
        torch.save(checkpoint, snapshot_path)
        print(f"  Saved training snapshot to: {snapshot_path}")

        # 3. 始终覆盖一个 "latest" 检查点，用于方便地恢复训练
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

    def restore(self, checkpoint_path: str):
        """
        从指定的检查点文件加载完整的训练状态
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return False

        print(f"  Loading training state from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型权重
            self.policy.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.policy.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.policy.actor_gnn.load_state_dict(checkpoint['actor_gnn_state_dict'])
            self.policy.critic_gnn.load_state_dict(checkpoint['critic_gnn_state_dict'])

            # 加载优化器状态 (非常重要!)
            self.policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            # 加载价值归一化器状态
            if self.trainer.value_normalizer is not None and 'value_normalizer_state_dict' in checkpoint:
                self.trainer.value_normalizer.load_state_dict(checkpoint['value_normalizer_state_dict'])
            
            # 获取上一次的 episode 编号，以便计算新的总步数和学习率衰减
            start_episode = checkpoint.get('episode', -1) + 1
            print(f"  Successfully loaded checkpoint. Resuming from episode {start_episode}.")
            return start_episode

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return -1 # 返回一个错误标识


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