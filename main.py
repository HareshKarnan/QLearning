import gym
from policy import QLearning

epsilon = 0.9
total_episodes = 500
max_steps = 1000

lr_rate = 0.81
gamma = 0.96

env = gym.make('FrozenLake-v0')
agent = QLearning(env,max_steps=max_steps,epsilon=epsilon,lr_rate=lr_rate)

agent.learn(total_episodes=total_episodes)
agent.get_q_values()
agent.plot_fig()


# learning is done


