from envs.migratory_pgg_env import MigratoryPGGEnv
from agents.agent import IndependentQLearningAgent

def train():
    env = MigratoryPGGEnv(num_agents=5, num_resources=3)
    agents = [IndependentQLearningAgent(state_size=env.num_resources, action_size=5) for _ in range(env.num_agents)]

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            actions = [agent.choose_action(state) for agent in agents]
            next_state, rewards = env.step(actions)
            for i, agent in enumerate(agents):
                agent.update_q_table(state, actions[i], rewards[i], next_state)
            state = next_state

if __name__ == "__main__":
    train() 