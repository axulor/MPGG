# graph_mpe_runner_v2.py (修改 eval 函数)

@torch.no_grad()
def eval(self, total_num_steps: int):
    """执行策略评估，运行更长的时间以观察长期行为。"""
    if self.eval_envs is None: return
    print(f"--- 开始评估 (在 {total_num_steps} 步) ---")

    # 定义评估的总步数，而不是回合数
    eval_total_steps = 10000 # 例如，评估 10000 步
    # 或者可以定义评估的总回合数，但每个回合跑满 max_cycles
    # eval_num_episodes = 5
    # eval_max_steps_per_episode = self.all_args.max_cycles # 每个评估回合跑满环境上限

    all_episode_rewards = []
    all_episode_coop_rates = []
    # 可以记录更详细的数据，例如每一步的合作率
    step_coop_rates = []

    # --- 由于只用单线程评估，简化逻辑 ---
    eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()
    eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
    eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

    current_eval_steps = 0
    current_episode_reward_sum = np.zeros((1, self.num_agents, 1), dtype=np.float32)
    current_episode_length = 0

    while current_eval_steps < eval_total_steps:
        self.trainer.prep_rollout()
        # --- 获取确定性动作 ---
        eval_action, eval_rnn_states_next = self.trainer.policy.act(
            eval_obs, # 直接使用，因为只有一个线程
            eval_node_obs,
            eval_adj,
            eval_agent_id,
            eval_rnn_states,
            eval_masks,
            deterministic=True,
        )
        eval_rnn_states = _t2n(eval_rnn_states_next) # 更新 RNN 状态
        eval_actions = _t2n(eval_action)

        # --- 转换动作为环境格式 ---
        eval_actions_env = self.convert_actions_for_env(eval_actions.reshape(1, self.num_agents, -1))[0] # Reshape -> Convert -> Reshape back

        # --- 与评估环境交互 ---
        (
            next_eval_obs, next_eval_agent_id, next_eval_node_obs, next_eval_adj,
            eval_rewards, eval_dones, eval_infos,
        ) = self.eval_envs.step(eval_actions_env)

        # --- 记录指标 ---
        current_episode_reward_sum += eval_rewards[0] # 累加当前回合奖励
        strategies = [eval_infos[0][a].get("strategy", np.nan) for a in range(self.num_agents)]
        if not np.isnan(strategies).all():
             step_coop_rates.append(np.nanmean(strategies))

        # --- 更新状态 ---
        eval_obs = next_eval_obs
        eval_agent_id = next_eval_agent_id
        eval_node_obs = next_eval_node_obs
        eval_adj = next_eval_adj

        current_eval_steps += 1
        current_episode_length += 1

        # --- 处理环境内部的终止 (达到 max_cycles) ---
        done_env = np.all(eval_dones[0]) # 检查是否整个环境都结束了
        if done_env or current_episode_length >= self.all_args.max_cycles: # 如果环境完成或达到最大步数
            all_episode_rewards.append(np.sum(current_episode_reward_sum)) # 记录整个回合的总奖励
            # 计算该回合的平均合作率 (如果需要)
            # all_episode_coop_rates.append(np.nanmean(step_coop_rates[-current_episode_length:]))

            # 重置 RNN 状态和累加器
            eval_rnn_states = np.zeros_like(eval_rnn_states)
            eval_masks = np.ones_like(eval_masks)
            current_episode_reward_sum = np.zeros_like(current_episode_reward_sum)
            current_episode_length = 0
            # 注意：环境应该会在 step 之后自动 reset (DummyVecEnv 的特性)
            # 如果环境不自动 reset，需要在这里手动调用 self.eval_envs.reset()
        else:
            # 如果环境未结束，更新 mask (通常为 1)
            eval_masks = np.ones_like(eval_masks)


    # --- 计算最终评估平均值 ---
    mean_eval_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
    # 计算整个评估期间的平均合作率
    mean_eval_coop_rate = np.nanmean(step_coop_rates) if step_coop_rates else 0

    print("-" * 30)
    print(f"评估完成 ({current_eval_steps} 步):")
    print(f"  平均回合总奖励 (基于环境重置): {mean_eval_reward:.3f}")
    print(f"  整体平均合作率: {mean_eval_coop_rate:.3f}")
    print("-" * 30)

    # --- 记录评估结果 ---
    eval_log_infos = {
        "eval/average_episode_rewards": mean_eval_reward,
        "eval/average_cooperation_rate": mean_eval_coop_rate
    }
    self.log_env(eval_log_infos, total_num_steps)