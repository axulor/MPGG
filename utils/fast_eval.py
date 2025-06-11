import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm

def _t2n(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None: return None
    return x.detach().cpu().numpy()

class FastEvaluate:
    def __init__(self,
                all_args: Any, 
                policy: Any,   
                eval_envs: Any, 
                run_dir: Path,
                ):
        print(f"--- Initializing FastEvaluate Class ---")
        self.all_args = all_args
        self.policy = policy # MARL policy, can be None if only evaluating baselines initially
        self.eval_envs = eval_envs
        
        if not isinstance(run_dir, Path):
            run_dir = Path(run_dir)
            
        self.plot_save_dir = run_dir / "evaluation_plots"
        self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_agents = getattr(all_args, 'num_agents')
        self.n_eval_rollout_threads = getattr(all_args, 'n_eval_rollout_threads')
        self.eval_rounds = getattr(all_args, 'eval_rounds')
        self.eval_steps_per_round = getattr(all_args, 'eval_steps_per_round')
        
        if self.eval_envs.action_space and isinstance(self.eval_envs.action_space, list) and len(self.eval_envs.action_space) > 0:
            self.action_dim = self.eval_envs.action_space[0].shape[0]
        else:
            try:
                self.action_dim = self.eval_envs.action_space.shape[0]
            except:
                print("Warning: Could not determine action_dim from eval_envs.action_space. Defaulting to 2.")
                self.action_dim = 2

        # MODIFICATION: Initialize cache for baseline policy results
        self._baseline_results_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._baselines_evaluated = False # Flag to indicate if baselines have been run once

        print(f"  FastEvaluator Config: Rounds={self.eval_rounds}, Steps/Round={self.eval_steps_per_round}")
        print(f"  Evaluation plots will be saved to: {self.plot_save_dir}")
        print("--- FastEvaluate Class Initialized ---")

    def get_policy_names(self, evaluate_marl_policy: bool = True) -> List[str]:
        names = []
        if evaluate_marl_policy and self.policy is not None: # Only add "marl" if requested and policy exists
            names.append("marl")
        # Baselines are always considered, their evaluation might be cached
        names.extend(["random_walk", "static"])
        return names

    @torch.no_grad()
    def _run_one_policy_evaluation(self, policy_name: str) -> Dict[str, np.ndarray]:
        # (内容与你上一版代码中的 _run_one_policy_evaluation 完全相同，此处省略以减少重复)
        # ... (完整的评估单个策略的逻辑) ...
        all_round_coop_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)
        all_round_reward_trajectories = np.full((self.eval_rounds, self.eval_steps_per_round), np.nan, dtype=np.float32)

        if policy_name == "marl" and self.policy is not None and hasattr(self.policy, 'actor'):
            self.policy.actor.eval() 

        print(f"    Evaluating policy '{policy_name}':") 
        for r_idx in tqdm(range(self.eval_rounds), desc=f"      Policy '{policy_name}' Rounds", leave=False):
            obs, agent_id, node_obs, adj = self.eval_envs.reset() 
            for step_idx in range(self.eval_steps_per_round):
                actions_np_batch = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.action_dim), dtype=np.float32)
                if policy_name == "marl":
                    obs_input = np.concatenate(obs)
                    node_obs_input = np.concatenate(node_obs)
                    adj_input = np.concatenate(adj)
                    agent_id_input = np.concatenate(agent_id)
                    action_out = self.policy.act(obs_input, node_obs_input, adj_input, agent_id_input)
                    actions_np_flat = _t2n(action_out)
                    if actions_np_flat is None: 
                        actions_np_flat = np.random.rand(self.n_eval_rollout_threads * self.num_agents, self.action_dim) * 2 - 1
                    actions_np_batch = actions_np_flat.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
                elif policy_name == "random_walk":
                    for i in range(self.n_eval_rollout_threads):
                        for j in range(self.num_agents):
                            actions_np_batch[i, j] = self.eval_envs.action_space[j].sample()
                elif policy_name == "static":
                    actions_np_batch.fill(0.0)
                next_obs, next_agent_id, next_node_obs, next_adj, rewards, dones, infos = self.eval_envs.step(actions_np_batch)
                step_coop_rates_this_step: List[float] = []
                step_rewards_this_step: List[float] = [] 
                for thread_idx in range(self.n_eval_rollout_threads):
                    if infos[thread_idx] and len(infos[thread_idx]) > 0:
                        agent0_info = infos[thread_idx][0]
                        cr = agent0_info.get("step_cooperation_rate")
                        ar_manual = np.mean(rewards[thread_idx]) 
                        if cr is not None: 
                            step_coop_rates_this_step.append(cr)
                        if ar_manual is not None:
                            step_rewards_this_step.append(ar_manual)
                all_round_coop_trajectories[r_idx, step_idx] = np.mean(step_coop_rates_this_step) if step_coop_rates_this_step else np.nan
                all_round_reward_trajectories[r_idx, step_idx] = np.mean(step_rewards_this_step) if step_rewards_this_step else np.nan
                obs, agent_id, node_obs, adj = next_obs, next_agent_id, next_node_obs, next_adj
                if np.all(dones): 
                    break 
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
        """
        执行所有定义策略的评估。
        对于基准策略 (random_walk, static)，如果已经评估过，则从缓存读取。
        """
        all_policies_results = {}
        # MARL 策略总是需要重新评估，因为它会变化
        if self.policy is not None:
            print(f"  Re-evaluating MARL policy...")
            results_marl = self._run_one_policy_evaluation("marl")
            all_policies_results["marl"] = results_marl
        
        baseline_policy_names = ["random_walk", "static"]
        # 检查基准策略是否需要评估 (只在第一次或强制重新评估时)
        if not self._baselines_evaluated:
            print(f"  Evaluating baseline policies for the first time...")
            for policy_name in tqdm(baseline_policy_names, desc="  Evaluating Baselines", leave=True):
                results_baseline = self._run_one_policy_evaluation(policy_name)
                self._baseline_results_cache[policy_name] = results_baseline # 存入缓存
                all_policies_results[policy_name] = results_baseline
            self._baselines_evaluated = True # 标记基准已评估
        else:
            print(f"  Loading baseline policy results from cache...")
            for policy_name in baseline_policy_names:
                if policy_name in self._baseline_results_cache:
                    all_policies_results[policy_name] = self._baseline_results_cache[policy_name]
                else:
                    # 这种情况不应该发生，如果 _baselines_evaluated 为 True
                    print(f"Warning: Baseline '{policy_name}' was expected in cache but not found. Re-evaluating.")
                    results_baseline = self._run_one_policy_evaluation(policy_name)
                    self._baseline_results_cache[policy_name] = results_baseline
                    all_policies_results[policy_name] = results_baseline
                    
        return all_policies_results

    # plot_results 方法与你上一版代码基本相同，只需确保它能正确处理
    # all_policies_results 的结构 (即字典的字典)
    def plot_results(self, 
                     all_policies_results: Dict[str, Dict[str, np.ndarray]], 
                     eval_episode_num: Optional[int] = None 
                    ):
        # (内容与你上一版代码中的 plot_results 完全相同，此处省略以减少重复)
        # ... (完整的绘图逻辑) ...
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
        for policy_name, results in all_policies_results.items(): # Iterate through evaluated policies
            if not results: continue # Skip if a policy had no results (e.g. MARL if no policy)
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
        for policy_name, results in all_policies_results.items(): # Iterate through evaluated policies
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