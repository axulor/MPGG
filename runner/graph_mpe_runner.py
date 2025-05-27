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
from utils.eval import Evaluate  # 导入策略评估类


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
        self.use_centralized_V = self.all_args.use_centralized_V            # 是否使用中心化 Critic 布尔值 bool
        self.num_env_steps = self.all_args.num_env_steps                    # 整个训练过程将要运行的环境交互总次数 int
        self.episode_length = self.all_args.episode_length                  # Replay Buffer 中存储的时序轨迹的长度 int 
        self.n_rollout_threads = self.all_args.n_rollout_threads            # 用于数据搜集的并行环境实例数量 int
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 用于策略评估的并行环境实例数量 int
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay        # 是否在训练过程中线性衰减学习率 bool
        self.hidden_size = self.all_args.hidden_size                        # 神经网络隐藏层的大小/维度 int


        # 时间间隔参数
        self.save_interval = self.all_args.save_interval    # 每隔 save_interval 个回合，保存当前的 Actor 和 Critic 网络权重
        self.log_interval = self.all_args.log_interval      # 每隔 log_interval 个回合, 写入训练日志
        self.use_eval = self.all_args.use_eval              # 训练暂停，使用当前的策略在独立的评估环境中运行，并记录性能指标     
        self.eval_interval = self.all_args.eval_interval    # 每隔 eval_interval 个回合, 启动评估

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


        # --- 2. 初始化策略、训练器和 Buffer ---

        # 确定 Critic 输入的观测空间
        if self.use_centralized_V:
            share_observation_space = self.envs.share_observation_space[0] 
        else:
            share_observation_space = self.envs.observation_space[0]

        # 初始化策略网络 (Actor 和 Critic)
        print("  (Runner) 初始化策略网络...")

        # 图环境 Policy 初始化 (保持上次修正后的参数列表)
        self.policy = GR_MAPPOPolicy(
            self.all_args,                      # 1. args
            self.envs.observation_space[0],     # 2. obs_space
            share_observation_space,            # 3. cent_obs_space
            self.envs.node_observation_space[0],# 4. node_obs_space
            self.envs.edge_observation_space[0],# 5. edge_obs_space
            self.envs.action_space[0],          # 6. act_space
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
        # MODIFICATION START: Pass bad_mask_shape to buffer if we decide to store it
        # For now, just ensure other shapes are correct.
        # The buffer will need to handle bad_masks internally based on infos.
        self.buffer = GraphReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.node_observation_space[0],
            self.envs.agent_id_observation_space[0],
            self.envs.share_agent_id_observation_space[0],
            self.envs.adj_observation_space[0], 
            self.envs.action_space[0],
            # Add bad_mask_space if you define it, e.g., spaces.Discrete(2) or spaces.Box(0,1, (1,))
        )
        # MODIFICATION END

        # 实例化评估器
        self.evaluator = None
        if self.all_args.use_eval and self.eval_envs is not None:
            print("  (Runner) 初始化评估器...")
            try:
                self.evaluator = Evaluate(
                    all_args=self.all_args,          # 传递完整的配置对象
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
        print("  (Runner) Warmup 完成 ")

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print(f"  (Runner) 总训练步数: {self.num_env_steps}, 每回合步数: {self.episode_length}, 并行环境数: {self.n_rollout_threads}")
        print(f"  (Runner) 将运行 {episodes} 个学习更新周期 ")

        # 核心训练循环
        # last_obs, last_agent_id, last_node_obs, last_adj are not strictly needed here
        # if buffer.after_update() correctly copies the last_obs to obs[0]
        # and warmup initializes obs[0] correctly.
        # The buffer's internal `step` pointer handles the circularity.

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # Rollout: Collect data for self.episode_length steps
            for step in range(self.episode_length):             
                # self.collect(step) uses self.buffer.obs[self.buffer.step] (or the current logical step in buffer)
                values, actions, action_log_probs = self.collect() # MODIFICATION: collect no longer needs `step`
                
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions)
                
                # MODIFICATION START: Prepare bad_masks from infos
                # Assuming marl_env.py puts 'bad_mask_indicator' in each agent's info dict
                bad_masks = []
                for thread_info in infos: # infos is List[List[Dict]] (n_threads, n_agents)
                    thread_bad_masks = []
                    for agent_info in thread_info:
                        thread_bad_masks.append(agent_info.get('bad_mask_indicator', True))
                    bad_masks.append(thread_bad_masks)
                bad_masks = np.array(bad_masks, dtype=np.bool_) # Shape: (n_threads, n_agents)
                # MODIFICATION END

                data = (obs, agent_id, node_obs, adj,
                        rewards, dones, bad_masks, infos, # MODIFICATION: added bad_masks
                        values, actions, action_log_probs)
                self.insert(data)

            # Computations after collecting a full episode_length segment
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 计算 Return 和 Advantage...")
            self.compute()
            print("  (Runner) 计算完成.")

            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 开始训练网络...")
            train_infos = self.train()
            print(" (Runner) 训练完成.")
            
            # MODIFICATION: process_infos now takes dones from the *last step of the rollout*
            # `dones` here is from the very last envs.step() call in the inner loop
            env_infos_for_log = self.process_infos(infos, dones) 

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            print(f"--- {episode + 1}/{episodes} 结束 (总步数: {total_num_steps}) ---")
            if train_infos:
                print("  训练指标 (Train Infos):")
                for key, value in train_infos.items():
                    if isinstance(value, (float, int, np.number)):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, torch.Tensor) and value.numel() == 1:
                        print(f"    {key}: {value.item():.4f}")
            else:
                print("  训练指标 (Train Infos): 无")

            if env_infos_for_log:
                print("  环境指标 (Env Infos ):")
                for key, value in env_infos_for_log.items():
                    if isinstance(value, (float, int, np.number)):
                        print(f"    {key}: {value:.4f}")
            else:
                print("  环境指标 (Env Infos): 无")

            if (episode + 1) % self.save_interval == 0 or episode == episodes - 1:
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 保存模型...")
                self.save()
                print("  (Runner) 模型已保存.")

            if (episode + 1) % self.log_interval == 0:
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 记录日志...")
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos_for_log, total_num_steps)
                print("  (Runner) 日志记录完成.")

            if self.use_eval and ((episode + 1) % self.eval_interval == 0 or episode == episodes - 1):
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 执行评估...")
                self.eval(total_num_steps)
                print("  (Runner) 评估完成.")

        print("所有训练回合已完成. ")


    def warmup(self):
        """
        初始化 Replay Buffer 中的第一个时间步的数据 (s_0).
        This is called once at the very beginning of training.
        """
        obs, agent_id, node_obs, adj = self.envs.reset() # This resets all parallel envs
        
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
            share_agent_id = agent_id

        # Store initial observations at buffer.obs[0], etc.
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()
        
        # For s_0, the 'previous' done state (dones_env[0]) is False.
        self.buffer.dones_env[0] = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1),
            dtype=np.float32
        )
        # MODIFICATION START: Also initialize bad_masks[0] if buffer stores it
        # Assuming s_0 is never a "bad terminal" state for GAE purposes.
        # If buffer has self.bad_masks:
        # self.buffer.bad_masks[0] = np.ones(
        # (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        # ) # Or np.bool_ depending on buffer's dtype for bad_masks
        # MODIFICATION END

        self.buffer.step = 0 # Explicitly set buffer's circular pointer to 0


    @torch.no_grad() 
    # MODIFICATION: `collect` no longer needs `step` argument,
    # it will use `self.buffer.step` to get the current obs `s_t`
    def collect(self) -> Tuple[arr, arr, arr]:
        """
        使用当前策略网络收集动作 for s_t (which is at self.buffer.obs[self.buffer.step]).
        """
        self.trainer.prep_rollout()

        # Get s_t from the buffer's current position `self.buffer.step`
        # Note: self.buffer.share_obs[self.buffer.step] has shape (n_threads, n_agents, obs_dim)
        # We need to concatenate along the thread dimension for batch processing by the policy.
        share_obs_batch = np.concatenate(self.buffer.share_obs[self.buffer.step])
        obs_batch = np.concatenate(self.buffer.obs[self.buffer.step])
        node_obs_batch = np.concatenate(self.buffer.node_obs[self.buffer.step])
        adj_batch = np.concatenate(self.buffer.adj[self.buffer.step])
        agent_id_batch = np.concatenate(self.buffer.agent_id[self.buffer.step])
        share_agent_id_batch = np.concatenate(self.buffer.share_agent_id[self.buffer.step])

        value, action, action_log_prob = self.trainer.policy.get_actions(
            share_obs_batch,    
            obs_batch,
            node_obs_batch,
            adj_batch,
            agent_id_batch,     
            share_agent_id_batch,
        )
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads)) 
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        return values, actions, action_log_probs

    # MODIFICATION: `insert` now accepts `bad_masks`
    def insert(self, data: Tuple):
        """
        将一个时间步收集到的数据插入到 Replay Buffer 中
        """
        (   obs_t_plus_1, agent_id_t_plus_1, node_obs_t_plus_1, adj_t_plus_1,
            rewards_t, dones_t, bad_masks_t, infos_t, # dones_t, bad_masks_t are for s_t -> s_t+1 transition
            values_t, actions_t, action_log_probs_t # These are for s_t
        ) = data

        if self.use_centralized_V:
            share_obs_t_plus_1 = obs_t_plus_1.reshape(self.n_rollout_threads, -1)
            share_obs_t_plus_1 = np.expand_dims(share_obs_t_plus_1, 1).repeat(self.num_agents, axis=1)
            share_agent_id_t_plus_1 = agent_id_t_plus_1.reshape(self.n_rollout_threads, -1)
            share_agent_id_t_plus_1 = np.expand_dims(share_agent_id_t_plus_1, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs_t_plus_1 = obs_t_plus_1
            share_agent_id_t_plus_1 = agent_id_t_plus_1
        
        self.buffer.insert(
            share_obs_t_plus_1,
            obs_t_plus_1,
            node_obs_t_plus_1,
            adj_t_plus_1,
            agent_id_t_plus_1,
            share_agent_id_t_plus_1,
            actions_t,              
            action_log_probs_t,     
            values_t,               
            rewards_t,              
            dones_t,
            bad_masks_t # MODIFICATION: Pass bad_masks to buffer's insert method
        )


    @torch.no_grad()
    def compute(self):
        """计算 GAE 回报和优势 """
        self.trainer.prep_rollout()
        
        # Get V(s_L) from the policy for bootstrapping GAE
        # s_L is stored at self.buffer.obs[self.buffer.step] because `insert` increments `step`
        # AFTER storing s_{t+1} at new_step+1 and a_t, r_t at new_step.
        # So, after episode_length inserts, self.buffer.step points to where the *next* s_0 (from previous s_L) is.
        # The actual s_L (last state of the rollout segment) is at (self.buffer.step -1 + L) % L,
        # and its successor s_{L+1} (or s_0 of next virtual segment) is at self.buffer.step.
        # The obs for s_L (final state of rollout) is at self.buffer.share_obs[self.buffer.step]
        # because after_update copies share_obs[L] to share_obs[0] and step becomes 0,
        # OR if after_update is not called yet, step is L (or 0 if L % L).
        # More simply: the next_values are for the states stored at index self.buffer.episode_length (which is obs[L+1])
        # OR, if buffer.step is now 0 after a full rollout, it's self.buffer.obs[0] which contains s_{L+1}.
        # The buffer.compute_returns expects next_value for the state *after* the last state in the rewards/actions sequence.
        # The last r_t, a_t are at index self.episode_length-1. Their successor state s_L is at self.obs[self.episode_length].
        
        next_values_obs_idx = self.buffer.step # This is where s_{L+1} (or s_0 of next segment) is stored after a full rollout
                                               # or where s_t is if in middle of rollout.
                                               # For GAE, we need V(state_after_last_action_in_buffer)
                                               # The last action is at buffer.actions[episode_length-1].
                                               # The state it leads to is stored at buffer.obs[episode_length].

        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[self.buffer.episode_length]), # obs for s_L (successor of a_{L-1})
            np.concatenate(self.buffer.node_obs[self.buffer.episode_length]),
            np.concatenate(self.buffer.adj[self.buffer.episode_length]),
            np.concatenate(self.buffer.share_agent_id[self.buffer.episode_length]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self): # No change here
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update() 
        return train_infos

    # process_infos already takes `dones`, which is good.
    # No change needed for save, restore, eval, log_train, log_env based on this step.
    # ... (rest of the methods: save, restore, eval, process_infos, log_train, log_env remain the same for now)


    @torch.no_grad()
    def eval(self, total_num_steps: int):
        """
        执行策略评估，调用独立的评估器。
        """
        if self.evaluator is None:
            if self.eval_envs is None:
                print("警告：评估环境未初始化 (self.eval_envs is None)，跳过评估。")
            elif not self.all_args.use_eval:
                print("配置中禁用了评估 (self.all_args.use_eval is False)，跳过评估。")
            else:
                print("警告：评估器未成功初始化，跳过评估。")
            return

        print(f"--- 开始评估 (在训练总步数 {total_num_steps} 时) ---")
        # 执行评估并获取结果
        eval_results = self.evaluator.eval_policy() # 返回一个字典

        # 绘制并保存结果图
        if eval_results:
            self.evaluator.plot_results(eval_results)
            # 将评估结果记录到 TensorBoard (如果 writter 可用)
            self.evaluator.log_data(eval_results, writer=self.writter, total_num_steps=total_num_steps)
        else:
            print("评估未产生数据。")
        print(f"--- 评估结束 (在训练总步数 {total_num_steps} 时) ---")

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


    # 处理训练时的环境指标
    # 在 GMPERunner 类中
    def process_infos(self, infos: List[List[Dict]], dones: arr) -> Dict: # 新增 dones 参数
        """
        处理在Rollout Segment最后一步收集到的环境信息列表。
        提取关键指标，并计算在所有并行线程间的平均值。

        Args:
            infos (List[List[Dict]]): 形状 (n_rollout_threads, num_agents) 的字典列表，
                                        包含每个并行环境最后一个时间步的info。
            dones (arr): 形状 (n_rollout_threads, num_agents) 的布尔数组，
                        指示每个并行环境在最后一个时间步之后是否终止。

        Returns:
            Dict: 包含聚合后指标的字典，用于TensorBoard日志记录。
        """
        env_infos_to_log = {} # 用于存储最终要记录到TensorBoard的指标

        # --- 用于累积各个线程在Segment最后一步的指标 ---
        # 步级指标
        segment_end_step_coop_rates = []
        segment_end_step_avg_rewards = []
        
        # 如果线程在Segment最后一步刚好完成了一个逻辑回合，我们可以记录其回合统计
        completed_logic_episode_total_rewards = [] # 该逻辑回合的总社会奖励
        completed_logic_episode_lengths = []       # 该逻辑回合的长度
        # (合作率的计算需要逻辑回合的总合作数和总步数，或者marl_env在done时直接提供)

        num_threads = len(infos)
        if num_threads == 0:
            return env_infos_to_log

        for thread_id in range(num_threads):
            if self.num_agents > 0 and len(infos[thread_id]) > 0:
                # 以第一个智能体的info获取共享的步级指标
                # infos[thread_id] 是一个包含 num_agents 个字典的列表
                # infos[thread_id][0] 是该线程第一个智能体在Segment最后一步的info
                last_step_info_agent0 = infos[thread_id][0] 

                # 1. 提取并记录当前Segment最后一步的步级指标
                step_cr = last_step_info_agent0.get("step_cooperation_rate")
                if step_cr is not None:
                    segment_end_step_coop_rates.append(step_cr)

                step_ar = last_step_info_agent0.get("step_avg_reward")
                if step_ar is not None:
                    segment_end_step_avg_rewards.append(step_ar)

                # 2. 检查这个线程是否在当前Rollout Segment的最后一步刚好完成了一个逻辑回合
                # dones[thread_id] 是一个 (num_agents,) 的布尔数组
                if dones[thread_id].all(): # 如果这个线程的所有智能体都done了
                    # 这是一个逻辑回合的结束
                    
                    # a. 计算这个完成的逻辑回合的总社会奖励
                    # total_rewards_count 是从上次reset到当前(done)步的累积个体奖励
                    ep_thread_total_social_reward = 0
                    for agent_idx in range(self.num_agents):
                        if agent_idx < len(infos[thread_id]):
                            ep_thread_total_social_reward += infos[thread_id][agent_idx].get("total_rewards_count", 0)
                    completed_logic_episode_total_rewards.append(ep_thread_total_social_reward)

                    # b. 获取这个完成的逻辑回合的长度
                    # marl_env.py 的 info 中有 "current_episode_steps"
                    ep_len = last_step_info_agent0.get("current_episode_steps") #MODIFED: Will add this to marl_env.py
                    if ep_len is not None:
                        completed_logic_episode_lengths.append(ep_len)
                    
                    # c. (可选) 计算这个完成的逻辑回合的平均合作率
                    # 需要: sum over agents (cooperation_counts[agent]) / (num_agents * ep_len)
                    # 如果 marl_env.py 在 info 中直接提供了 "final_episode_coop_rate" 会更方便
                    # 否则，我们可以在这里计算：
                    if ep_len is not None and ep_len > 0:
                        total_coop_actions_in_ep = 0
                        for agent_idx in range(self.num_agents):
                            if agent_idx < len(infos[thread_id]):
                                total_coop_actions_in_ep += infos[thread_id][agent_idx].get("cooperation_counts", 0)
                        avg_coop_rate_for_ep = total_coop_actions_in_ep / (self.num_agents * ep_len)
                        # (这个变量没用到，因为我们下面会用 completed_episode_coop_rates,
                        # 但逻辑上可以这样计算，或者期望marl_env提供。为简化，我们下面会用segment_end_step_coop_rates)
                        # 如果要记录这个，需要添加到 completed_episode_coop_rates 列表中

            else: # num_agents <= 0 or len(infos[thread_id]) == 0
                print(f"警告: process_infos 在 thread {thread_id} 中没有找到足够的 agent info。")

        # --- 计算并记录平均指标 ---

        # 平均最后一步的步级指标
        if segment_end_step_coop_rates:
            env_infos_to_log["segment_last_step/mean_cooperation_rate"] = np.mean(segment_end_step_coop_rates)
        if segment_end_step_avg_rewards:
            env_infos_to_log["segment_last_step/mean_avg_reward"] = np.mean(segment_end_step_avg_rewards)

        # 平均那些在Segment末尾刚好完成的逻辑回合的指标
        if completed_logic_episode_total_rewards:
            env_infos_to_log["completed_episode/mean_total_reward"] = np.mean(completed_logic_episode_total_rewards)
        if completed_logic_episode_lengths:
            env_infos_to_log["completed_episode/mean_length"] = np.mean(completed_logic_episode_lengths)
        
        # 记录在这个Rollout Segment中，有多少个并行环境完成了至少一个逻辑回合
        env_infos_to_log["completed_episode/num_in_segment"] = len(completed_logic_episode_total_rewards)

        return env_infos_to_log


    def log_train(self, train_infos: Dict, total_num_steps: int):
        """ 记录训练信息到 TensorBoard """
        for key, value in train_infos.items():
            if isinstance(value, (int, float, np.number)):
                self.writter.add_scalar(f"train/{key}", value, total_num_steps)


    def log_env(self, env_infos: Dict, total_num_steps: int):
        """ 记录环境信息到 TensorBoard """
        for key, value in env_infos.items():
            if isinstance(value, (int, float, np.number)):
                self.writter.add_scalar(f"env/{key}", value, total_num_steps)