import pickle
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class QLearning():
    def __init__(self,env,gamma=0.96,epsilon=0.9,lr_rate=0.81,max_steps=1000):
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env
        env.reset()
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr_rate = lr_rate
        self.qlearning_rew = []
        self.qlearning_time = [0]
        self.max_steps = max_steps
        super(QLearning, self).__init__()

    def learn(self,total_episodes=500):
        self.qlearning_rew = []
        self.qlearning_time = [0]

        for episode in range(total_episodes):
            state = self.env.reset()
            t = 0
            print('episode : ', episode)
            rew = 0
            while t < self.max_steps:
                action = self.choose_action(state)
                state2, reward, done, info = self.env.step(action)
                self.update(state, state2, reward, action)
                state = state2
                t += 1
                if done:
                    break
                time.sleep(0.1)
                rew += reward
            self.qlearning_rew.append(rew)
            self.qlearning_time.append(t + self.qlearning_time[-1])

    def choose_action(self,state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def update(self,state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)


    def plot_fig(self,save=True):
        fig = plt.figure()
        plt.plot(self.qlearning_time[1:], self.qlearning_rew)
        plt.xlabel('timesteps', fontsize=18)
        plt.ylabel('cumulative reward', fontsize=18)
        plt.show()
        if save:
            fig.savefig('q_learning_FrozenLakev0.jpg')

    def get_q_values(self):
        print(self.Q)

    def save(self):
        with open("frozenLake_qTable.pkl", 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self):
        raise NotImplementedError
