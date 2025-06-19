import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm 

# 假设 _t2n 在 utils.util 中或在此处定义
# (如果你的 _t2n 在别处，确保导入路径正确)
# from utils.util import _t2n 
def _t2n(x: Optional[torch.Tensor]) -> Optional[np.ndarray]: # 允许 x 为 None
    if x is None: return None
    return x.detach().cpu().numpy()

class FastEvaluate:
    def __init__(self,
                all_args: Any, 
                policy: Any,   # MARL policy network (GR_MAPPOPolicy instance)
                eval_envs: Any, # 并行评估环境 (GraphSubprocVecEnv instance)
                run_dir: Path,
                ):
        print(f"--- Initializing FastEvaluate Class ---")
        self.all_args = all_args
        self.policy = policy 
        self.eval_envs = eval_envs
        
        # 获取设备信息，用于将 NumPy 数据转换为 Tensor
        # 假设 all_args 中有 device 属性，或者 policy 对象有 device 属性
        if hasattr(all_args, 'device') and isinstance(all_args.device, torch.device):
            self.device = all_args.device
        elif hasattr(policy, 'device') and isinstance(policy.device, torch.device): # policy 通常有 device
            self.device = policy.device
        else: # Fallback
            self.device = torch.device("cuda" if all_args.cuda and torch.cuda.is_available() else "cpu")
            print(f"Warning: Device not explicitly found in all_args or policy. Defaulting FastEvaluate device to {self.device}")

        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)
            
        self.plot_save_dir = run_dir / "evaluation_plots"
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_agents = getattr(all_args, 'num_agents')
        self.n_eval_rollout_threads = getattr(all_args, 'n_eval_rollout_threads') # 这是 M
        
        self.eval_rounds = getattr(all_args, 'eval_rounds')
        self.eval_steps_per_round = getattr(all_args, 'eval_steps_per_round')
        
        # 获取动作维度
        # 假设 eval_envs.action_space 是一个列表，每个元素是单个智能体的 Box 空间
        if self.eval_envs.action_space and isinstance(self.eval_envs.action_space, list) and len(self.eval_envs.action_space) > 0:
            self.action_dim = self.eval_envs.action_space[0].shape[0]
        else: # Fallback for single Box or other structures (less common for multi-agent)
            try:
                self.action_dim = self.eval_envs.action_space.shape[0]
            except:
                print(f"Warning: Could not determine action_dim from eval_envs.action_space. Type: {type(self.eval_envs.action_space)}. Defaulting to 2.")
                self.action_dim = 2 

        self._baseline_results_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._baselines_evaluated = False

        print(f"  FastEvaluator Config: Rounds={self.eval_rounds}, Steps/Round={self.eval_steps_per_round}, Device={self.device}")
        print(f"  Evaluation plots will be saved to: {self.plot_save_dir}")
        print(f"DEBUG: FastEvaluate INSTANCE CREATED with id: {id(self)}") # <--- 新增打印
        print("--- FastEvaluate Class Initialized ---")

    def get_policy_names(self) -> List[str]:
        names = []
        if self.policy is not None: 
            names.append("marl")
        names.extend(["random_walk", "static"])
        return list(set(names)) # 使用 set 确保唯一性

    @torch.no_grad() # 评估不需要梯度
    def _run_one_policy_evaluation(self, policy_name: str) -> Dict[str, np.ndarray]:

        if policy_name in ["random_walk", "static"]:
            print(f"DEBUG: FastEvaluate._run_one_policy_evaluation CALLED FOR {policy_name.upper()}")

        all_round_coop_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)
        all_round_reward_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)

        if policy_name == "marl" and self.policy is not None:
            # 设置 Policy 内部所有相关模块 (Actor, Critic, GNNs) 为评估模式
            if hasattr(self.policy, 'actor') and self.policy.actor is not None: self.policy.actor.eval()
            if hasattr(self.policy, 'critic') and self.policy.critic is not None: self.policy.critic.eval()
            if hasattr(self.policy, 'actor_gnn') and self.policy.actor_gnn is not None: self.policy.actor_gnn.eval()
            if hasattr(self.policy, 'critic_gnn') and self.policy.critic_gnn is not None: self.policy.critic_gnn.eval()
        
        print(f"    Evaluating policy '{policy_name}':")
        for r_idx in tqdm(range(self.eval_rounds), desc=f"      Policy '{policy_name}' Rounds", leave=False, ncols=100):
            
            # 从环境获取的是 NumPy 数组
            # obs_np: (M, N, D_obs)
            # agent_id_np: (M, N, 1)
            # node_obs_all_agents_np: (M, N, N_nodes, D_node_raw)
            # adj_all_agents_np: (M, N, N_nodes, N_nodes)
            obs_np, agent_id_np, node_obs_all_agents_np, adj_all_agents_np = self.eval_envs.reset() 
            
            # 为 GNN 准备 M 份独特的图数据 (仍然是 NumPy)
            # node_obs_M_unique_np 形状 (M, N_nodes, D_node_raw)
            unique_node_obs_np = node_obs_all_agents_np[:, 0, :, :] # 取每个线程的第一个智能体携带的全局图
            # adj_M_unique_np 形状 (M, N_nodes, N_nodes)
            unique_adj_np = adj_all_agents_np[:, 0, :, :]

            for step_idx in range(self.eval_steps_per_round):
                actions_to_env_np = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)

                if policy_name == "marl":
                    # --- 准备 GR_MAPPOPolicy.get_actions 的输入 (转换为 Tensors) ---
                    # 1. 扁平化的个体观测 (M*N, D_obs)
                    flat_obs_t = torch.from_numpy(np.concatenate(obs_np)).float().to(self.device)
                    
                    # 2. M份独特图的节点和邻接数据 (M, N_nodes, D_node) 和 (M, N_nodes, N_nodes)
                    node_obs_for_gnn_t = torch.from_numpy(unique_node_obs_np).float().to(self.device)
                    adj_for_gnn_t = torch.from_numpy(unique_adj_np).float().to(self.device)
                    
                    # 3. 扁平化的 agent_id 和 env_id 用于挑选 (M*N, 1)
                    flat_agent_id_t = torch.from_numpy(np.concatenate(agent_id_np)).long().to(self.device)
                    
                    env_indices = np.arange(self.n_eval_rollout_threads)
                    flat_env_id_t = torch.from_numpy(
                        np.repeat(env_indices, self.num_agents).reshape(-1, 1)
                    ).long().to(self.device)
                    # --- 输入准备完毕 ---
                    
                    # 调用 GR_MAPPOPolicy.get_actions (它现在只返回 actions, action_log_probs)
                    torch_actions, _ = self.policy.get_actions(
                        flat_obs_t,
                        node_obs_for_gnn_t,
                        adj_for_gnn_t,
                        flat_agent_id_t,
                        flat_env_id_t
                    )
                    actions_from_policy_flat_np = _t2n(torch_actions)
                    if actions_from_policy_flat_np is None: 
                        actions_from_policy_flat_np = np.random.rand(
                            self.n_eval_rollout_threads * self.num_agents, self.action_dim
                        ) * 2 - 1 # Fallback
                    
                    actions_to_env_np = actions_from_policy_flat_np.reshape(
                        self.n_eval_rollout_threads, self.num_agents, -1
                    )
                
                elif policy_name == "random_walk":
                    # self.eval_envs.action_space 是一个列表，取第一个作为代表
                    action_space_sample = self.eval_envs.action_space[0]
                    low = action_space_sample.low
                    high = action_space_sample.high
                    
                    # 定义期望的输出形状
                    target_shape = (self.n_eval_rollout_threads, self.num_agents, self.action_dim)
                    
                    # 一次性生成所有随机动作
                    actions_to_env_np = np.random.uniform(low=low, 
                                                          high=high, 
                                                          size=target_shape).astype(np.float32)
                
                elif policy_name == "static":
                    actions_to_env_np.fill(0.0)
                
                # 执行环境步骤
                next_obs_tuple = self.eval_envs.step(actions_to_env_np)
                next_obs_np, next_agent_id_np, next_node_obs_all_agents_np, next_adj_all_agents_np, rewards, dones, infos = next_obs_tuple
                
                # 为 GNN 准备下一轮的独特图输入 (NumPy)
                next_unique_node_obs_np = next_node_obs_all_agents_np[:, 0, :, :]
                next_unique_adj_np = next_adj_all_agents_np[:, 0, :, :]

                # --- 记录当前步骤的指标 (与之前相同) ---
                step_coop_rates_this_step: List[float] = []
                step_rewards_this_step: List[float] = [] 
                for thread_idx in range(self.n_eval_rollout_threads):
                    if infos[thread_idx] and len(infos[thread_idx]) > 0:
                        agent0_info = infos[thread_idx][0]
                        cr = agent0_info.get("step_cooperation_rate")
                        ar_manual = np.mean(rewards[thread_idx]) 
                        if cr is not None: step_coop_rates_this_step.append(cr)
                        if ar_manual is not None: step_rewards_this_step.append(ar_manual)
                
                all_round_coop_trajectories[r_idx, step_idx] = np.mean(step_coop_rates_this_step) if step_coop_rates_this_step else np.nan
                all_round_reward_trajectories[r_idx, step_idx] = np.mean(step_rewards_this_step) if step_rewards_this_step else np.nan
                # --- 指标记录结束 ---

                # 更新状态以供下一评估步骤使用 (仍然是 NumPy 数组)
                obs_np, agent_id_np = next_obs_np, next_agent_id_np
                unique_node_obs_np, unique_adj_np = next_unique_node_obs_np, next_unique_adj_np # 更新为 GNN 的输入
                
                if np.all(dones): 
                    break 
        
        # ... (计算均值和标准差曲线的逻辑不变) ...
        mean_coop_curve = np.nanmean(all_round_coop_trajectories, axis=0)
        std_coop_curve = np.nanstd(all_round_coop_trajectories, axis=0)
        mean_reward_curve = np.nanmean(all_round_reward_trajectories, axis=0)
        std_reward_curve = np.nanstd(all_round_reward_trajectories, axis=0)
        policy_eval_results = {
            "cooperation_rate_curve": mean_coop_curve,
            "cooperation_rate_std_curve": std_coop_curve,
            "mean_reward_curve": mean_reward_curve,
            "mean_reward_std_curve": std_reward_curve,
        }
        return policy_eval_results

    def eval_policy(self) -> Dict[str, Any]:
        all_policies_results: Dict[str, Dict[str, np.ndarray]] = {}
        
        policy_names_to_evaluate_this_run = self.get_policy_names() # 获取所有当前可评估的策略

        print(f"  Starting evaluation for policies: {policy_names_to_evaluate_this_run}...")
        
        # 先处理 MARL 策略 (如果存在)
        if "marl" in policy_names_to_evaluate_this_run:
            print(f"    Re-evaluating MARL policy...")
            results_marl = self._run_one_policy_evaluation("marl")
            all_policies_results["marl"] = results_marl

        # 再处理基准策略
        baseline_policy_names = ["random_walk", "static"]
        baselines_actually_ran_this_time = False # 标记本次调用是否实际运行了基准

        for policy_name in baseline_policy_names:
            if policy_name in policy_names_to_evaluate_this_run: # 确保它在可评估列表中
                if not self._baselines_evaluated or policy_name not in self._baseline_results_cache:
                    if not self._baselines_evaluated:
                         print(f"    Evaluating baseline policy '{policy_name}' (first time or cache miss)...")
                    else: # 标志为True但缓存中没有
                         print(f"    WARNING: Baseline '{policy_name}' expected in cache but not found! Re-evaluating.")
                    
                    results_baseline = self._run_one_policy_evaluation(policy_name)
                    self._baseline_results_cache[policy_name] = results_baseline
                    all_policies_results[policy_name] = results_baseline
                    baselines_actually_ran_this_time = True
                else:
                    print(f"    Loading baseline policy '{policy_name}' from cache.")
                    all_policies_results[policy_name] = self._baseline_results_cache[policy_name]
        
        # 只有当这次实际运行了基准评估后，才将 _baselines_evaluated 设为 True
        if baselines_actually_ran_this_time:
            self._baselines_evaluated = True
            print(f"  DEBUG: Baselines that were run have been cached. self._baselines_evaluated is now True.")
                    
        return all_policies_results

    def plot_results(self, 
                     all_policies_results: Dict[str, Dict[str, np.ndarray]], 
                     eval_episode_num: Optional[int] = None 
                    ):
        # ... (这个函数与你上一版完全相同，用于绘图，不需要修改) ...
        if not all_policies_results:
            print("FastEvaluate: No evaluation results to plot.")
            return

        plot_labels_x_axis = np.arange(1, self.eval_steps_per_round + 1)
        seed = getattr(self.all_args, 'seed', 'unknown')
        
        if eval_episode_num is not None:
            file_suffix = f"ep{eval_episode_num}_s{seed}"
            eval_id_for_title = f"Eval@{eval_episode_num}" 
        else:
            current_time_str = time.strftime("%Y%m%d-%H%M%S")
            file_suffix = f"t{current_time_str}_s{seed}"
            eval_id_for_title = f"Time {current_time_str}"

        colors = {'marl': 'tab:blue', 'random_walk': 'tab:orange', 'static': 'tab:green', 'default': 'tab:grey'}
        linestyles = {'marl': '-', 'random_walk': '--', 'static': ':', 'default': '-.'}
        policy_display_names = {'marl': 'MARL', 'random_walk': 'Random Walk', 'static': 'Static'}

        fig_coop, ax_coop = plt.subplots(figsize=(12, 7))
        has_coop_data = False
        for policy_name, results in all_policies_results.items(): 
            if not results: continue 
            mean_coop = results.get("cooperation_rate_curve")
            std_coop = results.get("cooperation_rate_std_curve")
            if mean_coop is not None and len(mean_coop) == self.eval_steps_per_round:
                if not np.all(np.isnan(mean_coop)):
                    ax_coop.plot(plot_labels_x_axis, mean_coop, 
                                 label=policy_display_names.get(policy_name, policy_name), 
                                 color=colors.get(policy_name, colors['default']), 
                                 linestyle=linestyles.get(policy_name, linestyles['default']), 
                                 linewidth=2)
                    if std_coop is not None and policy_name == "marl" and not np.all(np.isnan(std_coop)):
                        ax_coop.fill_between(plot_labels_x_axis, mean_coop - std_coop, mean_coop + std_coop, 
                                             color=colors.get(policy_name, colors['default']), alpha=0.15)
                    has_coop_data = True
        
        if has_coop_data:
            ax_coop.set_xlabel("Evaluation Steps", fontsize=14)
            ax_coop.set_ylabel("Average Cooperation Rate", fontsize=14)
            title_coop = f"Cooperation Rate (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, " \
                         f"Avg over {self.eval_rounds} Rnds, ID: {eval_id_for_title})"
            ax_coop.set_title(title_coop, fontsize=16)
            ax_coop.legend(fontsize=12)
            ax_coop.grid(True, linestyle='--', alpha=0.7)
            ax_coop.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            coop_plot_path = self.plot_save_dir / f"coop_rate_{file_suffix}.png"
            plt.savefig(coop_plot_path)
            print(f"Cooperation rate plot saved to: {coop_plot_path}")
        else:
            print("FastEvaluate: No valid cooperation rate data to plot.")
            plt.close(fig_coop)
        
        fig_reward, ax_reward = plt.subplots(figsize=(12, 7))
        has_reward_data = False
        for policy_name, results in all_policies_results.items(): 
            if not results: continue
            mean_reward = results.get("mean_reward_curve")
            std_reward = results.get("mean_reward_std_curve")
            if mean_reward is not None and len(mean_reward) == self.eval_steps_per_round:
                if not np.all(np.isnan(mean_reward)):
                    ax_reward.plot(plot_labels_x_axis, mean_reward, 
                                   label=policy_display_names.get(policy_name, policy_name), 
                                   color=colors.get(policy_name, colors['default']), 
                                   linestyle=linestyles.get(policy_name, linestyles['default']), 
                                   linewidth=2)
                    if std_reward is not None and policy_name == "marl" and not np.all(np.isnan(std_reward)):
                        ax_reward.fill_between(plot_labels_x_axis, mean_reward - std_reward, mean_reward + std_reward, 
                                               color=colors.get(policy_name, colors['default']), alpha=0.15)
                    has_reward_data = True

        if has_reward_data:
            ax_reward.set_xlabel("Evaluation Steps", fontsize=14)
            ax_reward.set_ylabel("Average Mean Reward (per agent)", fontsize=14)
            title_reward = f"Mean Reward (N={self.num_agents}, Eval Steps={self.eval_steps_per_round}, " \
                           f"Avg over {self.eval_rounds} Rnds, ID: {eval_id_for_title})"
            ax_reward.set_title(title_reward, fontsize=16)
            ax_reward.legend(fontsize=12)
            ax_reward.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            reward_plot_path = self.plot_save_dir / f"mean_reward_{file_suffix}.png"
            plt.savefig(reward_plot_path)
            print(f"Mean reward plot saved to: {reward_plot_path}")
        else:
            print("FastEvaluate: No valid mean reward data to plot.")
            plt.close(fig_reward)