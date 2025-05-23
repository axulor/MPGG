import time
import numpy as np
from numpy import ndarray as arr # 类型别名，方便书写
from typing import Tuple, Dict, List 
import torch
import os

# from torch.utils.tensorboard import SummaryWriter # 可以用这个或 tensorboardX
from tensorboardX import SummaryWriter # 使用 tensorboardX
from algorithms.graph_mappo import GR_MAPPO 
from algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
from utils.graph_buffer import GraphReplayBuffer #  导入 Buffer 类


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
        # --- 1. 参数初始化 ---
        self.all_args = config["all_args"]      # 存储所有超参数的配置对象, types.SimpleNamespace
        self.envs = config["envs"]              # 训练环境, envs.env_wrappers.GraphDummyVecEnv 
        self.eval_envs = config["eval_envs"]    # 评估环境, envs.env_wrappers.GraphDummyVecEnv
        self.device = config["device"]          # 计算设备, torch.device 

        # 训练设置参数
        self.num_agents = self.all_args.num_agents                          # 环境中由策略控制的智能体数量 int
        self.use_centralized_V = self.all_args.use_centralized_V            # 是否使用中心化 Critic 布尔值 bool
        self.num_env_steps = self.all_args.num_env_steps                    # 整个训练过程将要运行的环境交互总次数的上限 int
        self.episode_length = self.all_args.episode_length                  # Replay Buffer 中存储的时序轨迹的长度 int 
        self.n_rollout_threads = self.all_args.n_rollout_threads            # 用于数据搜集的并行环境实例数量 int
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads  # 用于策略评估的并行环境实例数量 int
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay        # 是否在训练过程中线性衰减学习率 bool
        self.hidden_size = self.all_args.hidden_size                        # 神经网络隐藏层的大小/维度 int


        # 时间间隔参数
        self.save_interval = self.all_args.save_interval    # 每隔 save_interval 个回合，保存当前的 Actor 和 Critic 网络权重
        self.use_eval = self.all_args.use_eval              # 训练暂停，使用当前的策略在独立的评估环境中运行，并记录性能指标     
        self.eval_interval = self.all_args.eval_interval    # 每隔 eval_interval 个回合, 启动评估
        self.log_interval = self.all_args.log_interval      # 每隔 log_interval 个回合, 写入训练日志

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
        self.policy = Policy(
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
        )


    def run(self):
        """主训练循环"""
        print("  (Runner) 开始 Warmup...")
        self.warmup() # 初始化 Buffer 中的第一个时间步数据
        print("  (Runner) Warmup 完成 ")

        start_time = time.time() # 记录开始时间
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads # 计算总共需要运行的回合数
        print(f"  (Runner) 总训练步数: {self.num_env_steps}, 每回合步数: {self.episode_length}, 并行环境数: {self.n_rollout_threads}")
        print(f"  (Runner) 将运行 {episodes} 个回合 ")

        # --- 核心训练循环 ---
        for episode in range(episodes):
            # 学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # --- Rollout：注意 episode_length 仅仅是数据收集和网络更新的周期 ---
            for step in range(self.episode_length):
                # 1. 采样
                values, actions, action_log_probs, actions_env = self.collect(step)
                # 2. 交互
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions_env)
                # 3. 将数据插入 Buffer
                data = (obs, agent_id, node_obs, adj,
                        rewards, dones, infos,
                        values, actions, action_log_probs,
                        )
                self.insert(data)

            # --- 计算回报和优势 ---
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 计算 Return 和 Advantage...")
            self.compute()
            print("  (Runner) 计算完成.")

            # --- 训练网络 ---
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 开始训练网络...")
            train_infos = self.train()            # 返回训练 Infos
            print(" (Runner) 训练完成.")
            env_infos = self.process_infos(infos) # 处理环境 infos

            # --- 后处理与记录 ---
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads # 目前所在的总环境步长

            # === 打印回合结果 ===
            print(f"--- 回合 {episode + 1}/{episodes} 结束 (总步数: {total_num_steps}) ---")
            # 打印训练信息
            if train_infos: # 确保 train_infos 不是 None 或空
                print("  训练指标 (Train Infos):")
                for key, value in train_infos.items():
                    if isinstance(value, (float, int, np.number)): # 只打印标量值
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, torch.Tensor) and value.numel() == 1: # 单元素 Tensor
                        print(f"    {key}: {value.item():.4f}")
            else:
                print("  训练指标 (Train Infos): 无")

            # 打印环境信息 (来自 process_infos)
            if env_infos: # 确保 env_infos 不是 None 或空
                print("  环境指标 (Env Infos - 来自 process_infos):")
                for key, value in env_infos.items():
                    if isinstance(value, (float, int, np.number)):
                        print(f"    {key}: {value:.4f}")
            else:
                print("  环境指标 (Env Infos): 无")
            # ==========================

            # 保存模型
            if (episode + 1) % self.save_interval == 0 or episode == episodes - 1: # 达到保存间隔或者 episode 结束
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 保存模型...")
                self.save()
                print("  (Runner) 模型已保存.")

            # 记录日志
            if (episode + 1) % self.log_interval == 0: # 达到日志保存间隔
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 记录日志...")
                # 写入 TensorBoard
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                print("  (Runner) 日志记录完成.")

            #TODO 达到评估间隔或者 episode 结束, 执行评估
            if self.use_eval and ((episode + 1) % self.eval_interval == 0 or episode == episodes - 1):
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 执行评估...")
                self.eval(total_num_steps)
                print("  (Runner) 评估完成.")

        # 训练循环结束
        print("所有训练回合已完成。")


    def warmup(self):
        """
        初始化 Replay Buffer 中的第一个时间步的数据 
        """
        # 1. 重置环境，获取初始状态
        obs, agent_id, node_obs, adj = self.envs.reset()

        
        # print(f"obs:\ttype={type(obs)}, length={len(obs)}")
        # print(f"obs[0]:\ttype={type(obs[0])}, shape={obs[0].shape}, dtype={obs[0].dtype}")
        # print(obs)

        # print(f"agent_id:\ttype={type(agent_id)}, length={len(agent_id)}")
        # print(f"agent_id[0]:\ttype={type(agent_id[0])}, shape={agent_id[0].shape}, dtype={agent_id[0].dtype}")
        # print(agent_id)

        # print(f"node_obs:\ttype={type(node_obs)}, length={len(node_obs)}")
        # print(f"node_obs[0]:\ttype={type(node_obs[0])}, shape={node_obs[0].shape}, dtype={node_obs[0].dtype}")
        # print(node_obs)

        # print(f"adj:\ttype={type(adj)}, length={len(adj)}")
        # print(f"adj[0]:\ttype={type(adj[0])}, shape={adj[0].shape}, dtype={adj[0].dtype}")
        # print(adj)
        

        # 2. 准备中心化观测 (Share Observation)
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1) # 最终维度为 (1, N, N*D)，1表示线程数, N表示智能体数, D表示特征数
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1) # 最终维度为 (1, N, N), 第二个N表示ID数
        else:
            share_obs = obs # (N,D), N代表智能体个数
            share_agent_id = agent_id # (N,1), 1表示id向量维度为1

        # print(f"share_obs:\ttype={type(share_obs)}, shape={share_obs.shape}, dtype={share_obs.dtype}")
        # print(f"share_obs[0]:\ttype={type(share_obs[0])}, shape={share_obs[0].shape}, dtype={share_obs[0].dtype}")

        # print(f"obs:\ttype={type(obs)}, shape={obs.shape}, dtype={obs.dtype}")
        # print(f"obs[0]:\ttype={type(obs[0])}, shape={obs[0].shape}, dtype={obs[0].dtype}")

        # print(f"node_obs:\ttype={type(node_obs)}, shape={node_obs.shape}, dtype={node_obs.dtype}")
        # print(f"node_obs[0]:\ttype={type(node_obs[0])}, shape={node_obs[0].shape}, dtype={node_obs[0].dtype}")

        # print(f"adj:\ttype={type(adj)}, shape={adj.shape}, dtype={adj.dtype}")
        # print(f"adj[0]:\ttype={type(adj[0])}, shape={adj[0].shape}, dtype={adj[0].dtype}")


        # print(f"agent_id:\ttype={type(agent_id)}, shape={agent_id.shape}, dtype={agent_id.dtype}")
        # print(f"agent_id[0]:\ttype={type(agent_id[0])}, shape={agent_id[0].shape}, dtype={agent_id[0].dtype}")

        # print(f"share_agent_id:\ttype={type(share_agent_id)}, shape={share_agent_id.shape}, dtype={share_agent_id.dtype}")
        # print(f"share_agent_id[0]:\ttype={type(share_agent_id[0])}, shape={share_agent_id[0].shape}, dtype={share_agent_id[0].dtype}")
        # 3. 将初始状态存入 Buffer 的第 0 步
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()


    @torch.no_grad() 
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        """
        使用当前策略网络收集动作

        Args:
            step (int): 当前在 Replay Buffer 中的时间步索引

        Returns:
            Tuple[arr, arr, arr, arr, arr, arr]: 包含价值、动作、logp、RNN状态和环境动作
        """
        self.trainer.prep_rollout() #TODO 设置网络为评估模式

        # 从 Buffer 获取信息
        share_obs_batch = np.concatenate(self.buffer.share_obs[step]) # np.concatenate(...） 把这 T 个线程的数据沿第一个维度拼到一起, (T·N, N·D)
        obs_batch = np.concatenate(self.buffer.obs[step])
        node_obs_batch = np.concatenate(self.buffer.node_obs[step])
        adj_batch = np.concatenate(self.buffer.adj[step])
        agent_id_batch = np.concatenate(self.buffer.agent_id[step])
        share_agent_id_batch = np.concatenate(self.buffer.share_agent_id[step])

        # --- 调用策略网络 ---
        value, action, action_log_prob = self.trainer.policy.get_actions(
            share_obs_batch,    
            obs_batch,
            node_obs_batch,
            adj_batch,
            agent_id_batch,     
            share_agent_id_batch,
        )

        # print(f"value:\ttype={type(value)}, shape={value.shape}, dtype={value.dtype}")
        # print(f"value[0]:\ttype={type(value[0])}, shape={value[0].shape}, dtype={value[0].dtype}")

        # print(f"action:\ttype={type(action)}, shape={action.shape}, dtype={action.dtype}")
        # print(f"action[0]:\ttype={type(action[0])}, shape={action[0].shape}, dtype={action[0].dtype}")

        # print(f"action_log_prob:\ttype={type(action_log_prob)}, shape={action_log_prob.shape}, dtype={action_log_prob.dtype}")
        # print(f"action_log_prob[0]:\ttype={type(action_log_prob[0])}, shape={action_log_prob[0].shape}, dtype={action_log_prob[0].dtype}")



        # --- 处理网络输出 ---
        values = np.array(np.split(_t2n(value), self.n_rollout_threads)) # 加上并行线程数 T 维度
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads)) # 重新构造一个 (T, …) 的数组

        # print(f"values:\ttype={type(values)}, shape={values.shape}, dtype={values.dtype}")
        # print(f"values[0]:\ttype={type(values[0])}, shape={values[0].shape}, dtype={values[0].dtype}")

        # print(f"actions:\ttype={type(actions)}, shape={actions.shape}, dtype={actions.dtype}")
        # print(f"actions[0]:\ttype={type(actions[0])}, shape={actions[0].shape}, dtype={actions[0].dtype}")

        # print(f"action_log_probs:\ttype={type(action_log_probs)}, shape={action_log_probs.shape}, dtype={action_log_probs.dtype}")
        # print(f"action_log_probs[0]:\ttype={type(action_log_probs[0])}, shape={action_log_probs[0].shape}, dtype={action_log_probs[0].dtype}")

        # i = 6
        # v = value[i,0].item()
        # a = action[i,0].item()
        # lp = action_log_prob[i,0].item()
        # print(f"Agent {i} → value={v:.4f}, action={a}, log_prob={lp:.4f}")

        # i = 7
        # v = value[i,0].item()
        # a = action[i,0].item()
        # lp = action_log_prob[i,0].item()
        # print(f"Agent {i} → value={v:.4f}, action={a}, log_prob={lp:.4f}")

        # i = 8
        # v = value[i,0].item()
        # a = action[i,0].item()
        # lp = action_log_prob[i,0].item()
        # print(f"Agent {i} → value={v:.4f}, action={a}, log_prob={lp:.4f}")


        # --- 转换动作为环境格式 ---
        actions_env = None
        env_action_space = self.envs.action_space[0]
        # print(env_action_space.__class__.__name__)
        if env_action_space.__class__.__name__ == "Discrete":
            # 离散动作做 one-hot 编码
            action_indices = actions.astype(int) 
            if actions.ndim == 3 and actions.shape[-1] == 1:
                action_indices = action_indices.squeeze(-1)
            num_actions = env_action_space.n
            actions_env = np.eye(num_actions)[action_indices] # One-hot 编码
            # print(f"actions_env:\ttype={type(actions_env)}, shape={actions_env.shape}, dtype={actions_env.dtype}")
            # print(f"actions_env[0]:\ttype={type(actions_env[0])}, shape={actions_env[0].shape}, dtype={actions_env[0].dtype}")
        elif env_action_space.__class__.__name__ == "Box":
            # 连续动作直接传递
            actions_env = actions 
        else:
            print(f"错误: collect 中遇到未知的动作空间类型 {env_action_space.__class__.__name__}")
            raise NotImplementedError

        return values, actions, action_log_probs, actions_env


    def insert(self, data: Tuple):
        """
        将一个时间步收集到的数据插入到 Replay Buffer 中 (来自原 GMPERunner)。
        Args:
            data (Tuple): 包含单步数据的元组，顺序应为:
                        (obs, agent_id, node_obs, adj, rewards, dones, infos,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
        """
        # 解包数据
        (   obs, agent_id, node_obs, adj,
            rewards, dones, infos,
            values, actions, action_log_probs
        ) = data

        # TODO --- 处理 Masks ---
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = 0.0 # dones 为 True 的地方 mask 为 0

        # --- 构建中心化观测 ---
        if self.use_centralized_V:
            # 中性化观测相当于扩展了全局维度
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1) # (1, N, N*D)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1) # (1, N, N)
        else:
            share_obs = obs # (1, N*D)
            share_agent_id = agent_id # (1, N, N*D)

        # --- 调用 Buffer 的 insert 方法 ---
        # 关键: 确保参数顺序与 GraphReplayBuffer.insert 定义一致
        self.buffer.insert(
            share_obs,          # 下一步共享观测
            obs,                # 下一步个体观测
            node_obs,           # 下一步节点观测
            adj,                # 下一步邻接矩阵
            agent_id,           # 下一步智能体 ID
            share_agent_id,     # 下一步共享 ID
            actions,            # *上一步* 动作
            action_log_probs,   # *上一步* 动作 logp
            values,             # *上一步* 价值估计
            rewards,            # 当前步奖励
        )


    @torch.no_grad() # 不计算梯度
    def compute(self):
        """计算 GAE 回报和优势 """
        self.trainer.prep_rollout() # 设置网络为评估模式
        # 获取 Buffer 中最后一步状态的价值估计
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        # 调用 Buffer 计算 returns
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)


    @torch.no_grad() # TODO
    def eval(self, total_num_steps: int):
        """执行策略评估 """
        # 检查是否有评估环境
        if self.eval_envs is None:
            print("警告：未设置评估环境，跳过评估。")
            return

        eval_episode_rewards = [] # 记录每个评估回合的总奖励
        # --- 新增：记录合作率 ---
        eval_episode_cooperation_rates = []
        # --- 新增：记录其他可能的 MPGG 指标 ---
        # eval_episode_final_strategies = [] # 记录最终策略分布等

        # 重置评估环境
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        # 运行指定数量的评估回合
        num_eval_episodes_done = 0
        # 使用列表存储每个线程当前回合的数据
        episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        episode_strategies = [[] for _ in range(self.n_eval_rollout_threads)]

        while num_eval_episodes_done < self.all_args.eval_episodes:
            self.trainer.prep_rollout()

            # --- 获取确定性动作 ---
            # **修改:** 调用 policy.act 获取动作和 Actor RNN 状态
            #           Critic RNN 状态不需要在 act 中更新
            eval_action, eval_rnn_states_actor_next = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
            )
            # 更新 Actor RNN 状态
            # Critic RNN 状态在评估时不更新，保持为 0 或上一步状态（如果需要）

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))

            # --- 与评估环境交互 ---
            (
                eval_obs, eval_agent_id, eval_node_obs, eval_adj,
                eval_rewards, eval_dones, eval_infos,
            ) = self.eval_envs.step(eval_actions)

            # --- 记录当前步奖励和策略 ---
            for thread_id in range(self.n_eval_rollout_threads):
                episode_rewards[thread_id].append(eval_rewards[thread_id])
                thread_strategies = []
                for agent_id in range(self.num_agents):
                    if agent_id < len(eval_infos[thread_id]) and "strategy" in eval_infos[thread_id][agent_id]:
                        thread_strategies.append(eval_infos[thread_id][agent_id]["strategy"])
                    else:
                        thread_strategies.append(np.nan)
                episode_strategies[thread_id].append(thread_strategies)

            # --- 处理回合结束 ---
            eval_dones_env = np.all(eval_dones, axis=1)

            # --- 统计完成的回合 ---
            for i in range(self.n_eval_rollout_threads):
                if eval_dones_env[i]:
                    num_eval_episodes_done += 1
                    # 计算总奖励 (所有智能体、所有步骤奖励求和)
                    total_ep_reward = np.sum(np.array(episode_rewards[i]))
                    eval_episode_rewards.append(total_ep_reward)
                    # 计算平均合作率
                    ep_strategies = np.array(episode_strategies[i])
                    valid_strategies = ep_strategies[~np.isnan(ep_strategies)]
                    avg_coop_rate = np.mean(valid_strategies) if len(valid_strategies) > 0 else 0
                    eval_episode_cooperation_rates.append(avg_coop_rate)
                    # 清空记录
                    episode_rewards[i] = []
                    episode_strategies[i] = []
                    if num_eval_episodes_done >= self.all_args.eval_episodes: break
            if num_eval_episodes_done >= self.all_args.eval_episodes: break

        # --- 计算评估结果平均值 ---
        mean_eval_reward = np.mean(eval_episode_rewards) if eval_episode_rewards else 0
        mean_eval_coop_rate = np.mean(eval_episode_cooperation_rates) if eval_episode_cooperation_rates else 0

        print(f"  评估结果 ({num_eval_episodes_done} 回合):")
        print(f"    平均回合总奖励: {mean_eval_reward:.3f}")
        print(f"    平均合作率: {mean_eval_coop_rate:.3f}")

        # --- 记录评估结果 ---
        eval_env_infos = {
            "eval_average_episode_rewards": mean_eval_reward,
            "eval_average_cooperation_rate": mean_eval_coop_rate
        }
        self.log_env(eval_env_infos, total_num_steps)


    def train(self):
        """用 Buffer 中的数据训练策略 """
        self.trainer.prep_training() # 设置网络为训练模式
        train_infos = self.trainer.train(self.buffer) # 调用 trainer 的训练方法
        self.buffer.after_update() # 先采一整段轨迹 → 更新网络 → 再采新的一整段
        return train_infos

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


    # TODO
    def process_infos(self, infos: List[List[Dict]]) -> Dict:
        """
        处理环境返回的 info 列表，提取和聚合 MPGG 相关指标。
        这个函数在每个 rollout 结束时被调用，infos 是最后一个时间步的 info。
        因此，我们期望环境在最后一个时间步的 info 中包含了整个 episode 的聚合指标。

        Args:
            infos (List[List[Dict]]): 形状 (n_rollout_threads, num_agents) 的字典列表。
                                      每个字典包含了对应智能体在 episode 结束时的 info。

        Returns:
            Dict: 包含聚合后指标的字典，用于 TensorBoard 日志记录。
        """
        env_infos_to_log = {} # 用于存储最终要记录到 TensorBoard 的指标

        # --- 初始化用于累加所有线程的 episode 指标的列表 ---
        all_ep_social_gdp = []
        all_ep_avg_reward_per_agent_step = []
        all_ep_avg_cooperation_rate = []
        # 也可以收集个体累积奖励的分布等
        # all_ep_individual_cumulative_rewards = []

        num_threads = len(infos)
        if num_threads == 0:
            return env_infos_to_log

        for thread_id in range(num_threads):
            # 对于每个线程，我们通常只需要一个 agent 的 info 来获取 episode 级别的聚合指标，
            # 因为这些指标是在环境中计算的，并且对于该 episode 内的所有 agent 应该是相同的（如果是系统级指标）。
            # 或者，我们可以选择第一个 agent 的 info 作为代表。
            if self.num_agents > 0 and len(infos[thread_id]) > 0:
                # 以第一个 agent 的 info 为例，提取 episode 聚合指标
                # 假设这些 key 是在 marl_env.py 的 step 方法中，当 is_episode_done 时添加的
                first_agent_info_in_thread = infos[thread_id][0]

                ep_social_gdp = first_agent_info_in_thread.get("episode_social_gdp")
                if ep_social_gdp is not None:
                    all_ep_social_gdp.append(ep_social_gdp)

                ep_avg_reward = first_agent_info_in_thread.get("episode_avg_reward_per_agent_step")
                if ep_avg_reward is not None:
                    all_ep_avg_reward_per_agent_step.append(ep_avg_reward)

                ep_coop_rate = first_agent_info_in_thread.get("episode_avg_cooperation_rate")
                if ep_coop_rate is not None:
                    all_ep_avg_cooperation_rate.append(ep_coop_rate)

                # 如果需要记录每个 agent 的累积奖励，可以这样做：
                # for agent_id_local in range(self.num_agents):
                #     if agent_id_local < len(infos[thread_id]):
                #         agent_specific_info = infos[thread_id][agent_id_local]
                #         ind_cum_rew = agent_specific_info.get("episode_individual_cumulative_reward")
                #         if ind_cum_rew is not None:
                #             all_ep_individual_cumulative_rewards.append(ind_cum_rew)
            else:
                print(f"警告: process_infos 在 thread {thread_id} 中没有找到足够的 agent info。")


        # --- 计算所有线程的平均值 ---
        if all_ep_social_gdp:
            env_infos_to_log["episode/social_gdp_mean"] = np.mean(all_ep_social_gdp)
            # 也可以记录标准差等
            # env_infos_to_log["episode/social_gdp_std"] = np.std(all_ep_social_gdp)
        if all_ep_avg_reward_per_agent_step:
            env_infos_to_log["episode/avg_reward_per_agent_step_mean"] = np.mean(all_ep_avg_reward_per_agent_step)
        if all_ep_avg_cooperation_rate:
            env_infos_to_log["episode/avg_cooperation_rate_mean"] = np.mean(all_ep_avg_cooperation_rate)

        # --- (可选) 记录每一步的平均合作率和平均个体奖励 ---
        # 这需要从 buffer 中提取数据，或者让环境在每一步的 info 中提供
        # 这里仅作示例，假设我们想记录最后一个 rollout 中所有 agent 在所有 step 的平均合作率
        # 注意：buffer 中的数据是 (episode_length+1, n_threads, n_agents, ...)
        # infos 传入的是最后一个 step 的 info，可能不适合计算 rollout 平均值
        # 更稳妥的做法是在 Runner 的 run 循环中，在 self.insert(data) 之后，
        # 实时地从 infos 中提取每一步的合作信息并累加，然后在 log_interval 时计算平均值。
        # 但为了简化，这里我们只处理 episode 结束时的聚合信息。

        # 简单示例：记录最后一个 step 的系统平均合作率和奖励 (如果 env_infos 中有)
        # 这部分与之前的 process_infos 逻辑类似，但现在我们更关注 episode 级别的指标
        step_rewards = []
        step_strategies = []
        for thread_infos in infos:
            for agent_info in thread_infos:
                 if isinstance(agent_info, dict):
                    step_rewards.append(agent_info.get("individual_reward_step", np.nan))
                    step_strategies.append(agent_info.get("strategy_step", np.nan))

        valid_step_rewards = [r for r in step_rewards if not np.isnan(r)]
        valid_step_strategies = [s for s in step_strategies if not np.isnan(s)]

        if valid_step_rewards:
            env_infos_to_log["step/mean_reward"] = np.mean(valid_step_rewards)
        if valid_step_strategies:
            env_infos_to_log["step/cooperation_rate"] = np.mean(valid_step_strategies)


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