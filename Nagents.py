import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import itertools
import pandas as pd

style.use("ggplot")

SHOW_EVERY = 6000  # how often to play through env visually.


action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']


class Agent:
    def __init__(self, learning_rate=0.1, reward_decay=0.95, eps_greedy=0.9, eps_decay=0.9996):
        self.learning_rate = learning_rate
        # gamma, discount, reward decay
        self.gamma = reward_decay
        # epsilon
        self.eps_greedy = eps_greedy
        self.eps_decay = eps_decay
        # Q-table
        self.q_table = pd.DataFrame(0, index=[0], columns=action_space)

    def choose_action(self):
        if np.random.uniform() > self.eps_greedy:
            # only one observation/state
            # exploitation
            state_action = self.q_table.loc[0, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # exploration
            action = np.random.choice(action_space)
        return action

    def learn(self, action, reward):
        # update q_table
        current_q = self.q_table.loc[0, action]
        max_future_q = self.q_table.loc[0, :].max()
        new_q = current_q + self.learning_rate * ((reward + self.gamma * max_future_q) - current_q)
        self.q_table.loc[0, action] = new_q

        self.eps_greedy *= self.eps_decay


class DPG:
    def __init__(self, agents):
        self.mu = np.array([0, 0])
        self.sigma_square_s = np.array([0.5, 0.3])
        self.beta = np.array([5, -3])
        self.sigma_square = 1
        self.r_max = 15
        self.agents = agents
        # shapley or LOO
        self.pricing_mechanism = 'shapley'

        self.episode_rewards = []
        self.episode_rewards_sum = []

    def v(self, indices, y_tilde_s, y):
        sum_y_tilde_s = 0
        for i in indices:
            sum_y_tilde_s += y_tilde_s[i]
        return self.r_max - (sum_y_tilde_s - y) ** 2

    def shapley(self, y_tilde_s, y, j, n):
        p = 0
        for perm in itertools.permutations(range(n)):
            for i in perm:
                if perm[i] == j:
                    # print(f"perm[:i +1]: {perm[:i+1]}, perm[:i]: {perm[:i]}")
                    # print(f"p = {p}")
                    p += self.v(perm[:i + 1], y_tilde_s, y) - self.v(perm[:i], y_tilde_s, y)
                    # print(f"p = {p}, v(perm[:i+1]) = {self.v(perm[:i+1], y_tilde_s, y)}, v(perm[:i] = {self.v(perm[:i], y_tilde_s, y)}")
                    # print(f"p = {p/math.factorial(n)}")
        return p / math.factorial(n)

    def loo(self, y_tilde_s, y, j, n):
        x = list(range(n))
        x.remove(j)
        p = self.v(range(n), y_tilde_s, y) - self.v(x, y_tilde_s, y)
        return p

    def step(self, actions):
        n = len(actions)
        s_square_s = np.zeros(n)
        x_s = np.zeros(n)
        x_tilde_s = np.zeros(n)
        y_tilde_s = np.zeros(n)
        i: int
        for i in range(n):
            x_s[i] = np.random.normal(self.mu[i], self.sigma_square_s[i] ** .5)

            if actions[i] == '0':
                s_square_s[i] = self.sigma_square_s[i]
                x_tilde_s[i] = self.mu[i]
            else:
                s_square_s[i] = float(actions[i])
                x_tilde_s[i] = x_s[i] + np.random.normal(0, s_square_s[i] ** .5)

            y_tilde_s[i] = x_tilde_s[i] * self.beta[i]
            # y_tilde = sum(x_tilde_s * self.beta)

        epsilon = np.random.normal(0, self.sigma_square ** .5)

        y = sum(x_s * self.beta) + epsilon
        # Y = X1 * self.beta1 + X2 * self.beta2 + epsilon
        # TODO: testing
        #y_tilde_s = np.array([2.395, 1.338])
        #y = 2.459
        #s_square_s = np.array([0.1, 0.1])

        prices = np.zeros(n)
        costs = np.zeros(n)
        profits = np.zeros(n)
        for i in range(n):
            # print(f"y_tilde_s: {y_tilde_s}, y: {y}, i: {i}, n: {n}")
            if self.pricing_mechanism == 'shapley':
                prices[i] = self.shapley(y_tilde_s, y, i, n)
            elif self.pricing_mechanism == 'LOO':
                prices[i] = self.loo(y_tilde_s, y, i, n)
            costs[i] = (self.sigma_square_s[i] / s_square_s[i]) + math.log10(s_square_s[i]) - (1 + math.log10(self.sigma_square_s[i]))
            # print(f"player {i}, costs: {costs[i]}")
            profits = prices - costs
            # print(f"player {i}, profit: {profits[i]}")
        return profits
        # TODO: is this correct?

    def train(self, episodes=10000):
        # print(f"episodes: {episodes}")
        for episode in range(episodes):
            # print(f"episode: {episode}")
            actions = []
            for agent in self.agents:
                action = agent.choose_action()
                actions.append(action)

            rewards = self.step(actions)

            for j in range(len(agents)):
                # print(f"player {j} action: {actions[j]} profit: {rewards[j]}")
                self.agents[j].learn(actions[j], rewards[j])

            self.episode_rewards.append(rewards)
            episode_reward = sum(rewards)
            self.episode_rewards_sum.append(episode_reward)

            if episode % SHOW_EVERY == 0:
                show = True
                # print(f"Episode: #{episode}")
                print(f"{SHOW_EVERY} episode mean: {np.mean(self.episode_rewards_sum[-SHOW_EVERY:])}")
                for j in range(len(agents)):
                    print(f"player {j} epsilon: {agents[j].eps_greedy}")
                    print(f"player {j} action: {actions[j]} profit: {rewards[j]}")
                    print(f"player {j} q-table")
                    print(agents[j].q_table)
            else:
                show = False
        moving_avg_sum = np.convolve(self.episode_rewards_sum, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

        # moving_avg = []
        rewards_per_agent = list(zip(*self.episode_rewards))
        for j in range(len(self.agents)):
            moving_avg = np.convolve(rewards_per_agent[j], np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
            plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.plot([i for i in range(len(moving_avg_sum))], moving_avg_sum)

        plt.legend(['agent 1', 'agent 2', 'sum'], loc='upper left')
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        plt.xlabel("episode #")
        plt.show()

'''
if start_q_table1 is None:
    q_table1 = {}
    for i in range(SIZE):
        q_table1[i] = [np.random.uniform(-15, 15) for i in range(n_actions)]
else:
    with open(start_q_table1, "rb") as f:
        q_table1 = pickle.load(f)

if start_q_table2 is None:
    q_table2 = {}
    for i in range(SIZE):
        q_table2[i] = [np.random.uniform(-15, 15) for i in range(n_actions)]
else:
    with open(start_q_table2, "rb") as f:
        q_table2 = pickle.load(f)
'''

# plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
# plt.ylabel(f"Reward")
# plt.xlabel(f"episode #")
# plt.show()
'''
def save_q_table():
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table1, f)
'''

if __name__ == "__main__":
    # training
    p0 = Agent()
    p1 = Agent()
    agents = [p0, p1]

    dpg = DPG(agents)
    print("training ...")
    dpg.train(30000)
