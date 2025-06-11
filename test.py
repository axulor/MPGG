import numpy as np

episode_length = 126
n_rollout_threads = 8
num_agents = 100


share_obs = np.zeros((episode_length + 1, n_rollout_threads, num_agents, *share_obs_shape), dtype=np.float32)