import time
import numpy as np
from numpy import ndarray as arr # 类型别名，方便书写
from typing import Tuple, Dict 
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
        self.recurrent_N = self.all_args.recurrent_N                        # 循环神经网络的层数 int

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
        print("  (Runner) Warmup 完成.")

        start_time = time.time() # 记录开始时间
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads # 计算总共需要运行的回合数
        print(f"  (Runner) 总训练步数: {self.num_env_steps}, 每回合步数: {self.episode_length}, 并行环境数: {self.n_rollout_threads}")
        print(f"  (Runner) 将运行 {episodes} 个回合。")

        # --- 核心训练循环 ---
        for episode in range(episodes):
            # 学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # --- Rollout：注意 episode_length 仅仅是数据收集和网络更新的周期 ---
            for step in range(self.episode_length):
                # 1. 采样
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # 2. 交互
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions_env)
                # 3. 将数据插入 Buffer
                data = (obs, agent_id, node_obs, adj,
                        rewards, dones, infos,
                        values, actions, action_log_probs,
                        rnn_states, rnn_states_critic)
                self.insert(data)

            # --- 计算回报和优势 ---
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 计算 Return 和 Advantage...")
            self.compute()
            print("  (Runner) 计算完成.")

            # --- 训练网络 ---
            print(f"  (Runner) 回合 {episode + 1}/{episodes}: 开始训练网络...")
            train_infos = self.train() 
            print("  (Runner) 训练完成.")

            # --- 后处理与记录 ---
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads # 目前所在的总环境步长

            # 保存模型
            if (episode + 1) % self.save_interval == 0 or episode == episodes - 1: # 达到保存间隔或者 episode 结束
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 保存模型...")
                self.save()
                print("  (Runner) 模型已保存.")

            # 记录日志
            if (episode + 1) % self.log_interval == 0: # 达到日志保存间隔
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"  (Runner) 回合 {episode + 1}/{episodes}: 记录日志...")
                #TODO 处理环境 infos
                env_infos = self.process_infos(infos) 
                # 计算平均回合奖励估计
                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                print(
                    f"回合 [{episode + 1}/{episodes}] | "
                    f"总步数: {total_num_steps}/{self.num_env_steps} "
                    f"({total_num_steps / self.num_env_steps * 100:.1f}%) | "
                    f"平均回合奖励 (估计): {avg_ep_rew:.3f} | "
                    f"用时: {elapsed_time:.2f}s"
                )
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

        # 2. 准备中心化观测 (Share Observation)
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1) # 最终维度为 (1, N, N*D)，1表示线程数, N表示智能体数, D表示特征数
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1) # 最终维度为 (1, N, N), 第二个N表示ID数
        else:
            share_obs = obs # (N,D), N代表智能体个数
            share_agent_id = agent_id # (N,1), 1表示id向量维度为1

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

        rnn_states_actor_batch = np.concatenate(self.buffer.rnn_states[step])
        rnn_states_critic_batch = np.concatenate(self.buffer.rnn_states_critic[step])
        masks_batch = np.concatenate(self.buffer.masks[step])

        # --- 调用策略网络 ---
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            share_obs_batch,    # 作为 cent_obs 传入
            obs_batch,
            node_obs_batch,
            adj_batch,
            agent_id_batch,     # Actor 可能用
            share_agent_id_batch,# Critic 可能用
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            masks_batch,
            deterministic=False # 训练时采样
        )

        # --- 处理网络输出 ---
        values = np.array(np.split(_t2n(value), self.n_rollout_threads)) # 按并行线程数拆分成 T 段
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads)) # 重新构造一个 (T, …) 的数组

        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # --- 转换动作为环境格式 ---
        # TODO 这里似乎可以简化
        actions_env = None
        env_action_space = self.envs.action_space[0]
        if env_action_space.__class__.__name__ == "MultiDiscrete": 
            # 对每个子离散分支分别做 one-hot，然后拼接
            all_one_hot_parts = []
            num_parts = env_action_space.shape
            action_dims = env_action_space.high + 1
            for i in range(num_parts):
                action_part_indices = actions[:, :, i]
                dim_size = action_dims[i]
                one_hot_part = np.eye(dim_size)[action_part_indices]
                all_one_hot_parts.append(one_hot_part)
            actions_env = np.concatenate(all_one_hot_parts, axis=2)
        elif env_action_space.__class__.__name__ == "Discrete":
            # 单一离散分支做 one-hot
            action_indices = actions.astype(int) 
            if actions.ndim == 3 and actions.shape[-1] == 1:
                action_indices = action_indices.squeeze(-1)
            num_actions = env_action_space.n
            actions_env = np.eye(num_actions)[action_indices] # One-hot 编码
        elif env_action_space.__class__.__name__ == "Box":
            # 连续动作直接传递
            actions_env = actions 
        else:
            print(f"错误: collect 中遇到未知的动作空间类型 {env_action_space.__class__.__name__}")
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env


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
            values, actions, action_log_probs,
            rnn_states, rnn_states_critic,
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
            rnn_states,         # *上一步* Actor RNN 输出 (作为下一步输入)
            rnn_states_critic,  # *上一步* Critic RNN 输出 (作为下一步输入)
            actions,            # *上一步* 动作
            action_log_probs,   # *上一步* 动作 logp
            values,             # *上一步* 价值估计
            rewards,            # 当前步奖励
            masks               # 当前步 Mask (1 if not done else 0)
        )


    @torch.no_grad() # 不计算梯度
    def compute(self):
        """计算 GAE 回报和优势 (来自原 GMPERunner)。"""
        self.trainer.prep_rollout() # 设置网络为评估模式
        # 获取 Buffer 中最后一步状态的价值估计
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
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

        # 初始化评估用的 RNN 状态和 Mask
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        # **注意:** Critic RNN 状态也需要初始化
        eval_rnn_states_critic = np.zeros_like(eval_rnn_states, dtype=np.float32) # 假设形状相同

        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

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
                np.concatenate(eval_rnn_states), # 传入 Actor 的当前状态
                np.concatenate(eval_masks),
                deterministic=True,
            )
            # 更新 Actor RNN 状态
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states_actor_next), self.n_eval_rollout_threads))
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

            # 重置结束回合的 Actor RNN 状态和 Mask
            eval_rnn_states[eval_dones_env == True] = 0.0
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0.0

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
    def process_infos(self, infos: list) -> Dict: 
        """
        处理环境返回的 info 列表 (来自原 BaseRunner，已为 MPGG 简化)。
        提取个体奖励和策略，计算系统平均值
        这里需要进行大幅修改和重写

        Args:
            infos (list): 格式 List[List[Dict]] (n_threads, n_agents)。

        Returns:
            Dict: 包含聚合后信息的字典。
        """
        env_infos = {}
        num_threads = len(infos)
        if num_threads == 0 or len(infos[0]) == 0: return env_infos

        # 确保 infos 结构符合预期
        if not isinstance(infos[0], list) or not isinstance(infos[0][0], dict):
            print(f"警告：process_infos 收到意外的 infos 格式: {type(infos[0])}")
            return env_infos

        # 收集所有线程和智能体的指标
        all_rewards = []
        all_strategies = []
        for thread_id in range(num_threads):
            for agent_id in range(self.num_agents):
                if agent_id < len(infos[thread_id]):
                    agent_info = infos[thread_id][agent_id]
                    all_rewards.append(agent_info.get("individual_reward", np.nan))
                    all_strategies.append(agent_info.get("strategy", np.nan))
                else: # 处理可能的长度不匹配
                    all_rewards.append(np.nan)
                    all_strategies.append(np.nan)

        # 计算系统平均指标 (忽略 NaN)
        valid_rewards = [r for r in all_rewards if not np.isnan(r)]
        valid_strategies = [s for s in all_strategies if not np.isnan(s)]

        if valid_rewards:
            env_infos["system/mean_individual_reward"] = np.mean(valid_rewards)
        if valid_strategies:
            env_infos["system/cooperation_rate"] = np.mean(valid_strategies)

        # 也可以记录每个智能体的平均指标（如果需要）
        # for agent_id in range(self.num_agents):
        #     agent_rewards = [infos[t][agent_id].get("individual_reward", np.nan) for t in range(num_threads) if agent_id < len(infos[t])]
        #     valid_agent_rewards = [r for r in agent_rewards if not np.isnan(r)]
        #     if valid_agent_rewards:
        #          env_infos[f"agent{agent_id}/individual_reward_mean"] = np.mean(valid_agent_rewards)
        #     # ... (记录策略等) ...

        return env_infos


    def log_train(self, train_infos: Dict, total_num_steps: int):
        """记录训练信息到 TensorBoard (来自原 BaseRunner)。"""
        if self.writter is None: return # 如果未使用 TensorBoard 则跳过
        for key, value in train_infos.items():
            if isinstance(value, (int, float, np.number)):
                self.writter.add_scalar(f"train/{key}", value, total_num_steps)


    def log_env(self, env_infos: Dict, total_num_steps: int):
        """记录环境信息到 TensorBoard (来自原 BaseRunner)。"""
        if self.writter is None: return
        for key, value in env_infos.items():
            if isinstance(value, (int, float, np.number)):
                # key 可能已经是 "system/..." 或 "agentX/..."
                self.writter.add_scalar(f"env/{key}", value, total_num_steps)