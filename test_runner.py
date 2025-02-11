from envs.migratory_pgg_env import MigratoryPGGEnv
from algorithms.DQNPolicy import DQNPolicy
from utils.ReplayBuffer import ReplayBuffer
import numpy as np
import torch

class Runner:
    def __init__(self):
        self.env = MigratoryPGGEnv()
        self.policy = DQNPolicy(self.env.observation_space("agent_0"), self.env.action_space("agent_0"))
        self.buffer = ReplayBuffer(capacity=10000)

    def run(self, num_episodes=1000):
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = {agent: False for agent in self.env.agents}

            while not all(done.values()):
                actions = {agent: self.policy.select_action(obs[agent]) for agent in self.env.agents}
                next_obs, rewards, terminations, truncations, _ = self.env.step(actions)

                for agent in self.env.agents:
                    self.buffer.store(obs[agent], actions[agent], rewards[agent], next_obs[agent], terminations[agent])

                obs = next_obs
                done = {agent: terminations[agent] or truncations[agent]}

                if len(self.buffer) > 1000:
                    self.policy.train(self.buffer)

            print(f"Episode {episode} complete.")

if __name__ == "__main__":
    runner = Runner()
    runner.run()
