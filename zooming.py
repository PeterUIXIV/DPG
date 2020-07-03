import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import itertools
import pandas as pd
import portion as P

style.use("ggplot")

SHOW_EVERY = 500  # how often to play through env visually.
show = False
test_pricing = False


action_space = ['0', '0.025', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1']


class Agent:
    def __init__(self, beta, sigma_square, learning_rate=0.1, reward_decay=0.95, eps_greedy=0.95, eps_decay=0.9995,
                 mu=0, value_function='action_value_method'):
        # value functions: action_value_method, q-learning, UCB
        self.value_function = value_function

        self.mu = mu
        self.beta = beta
        self.sigma_square = sigma_square

        self.active_arms = []
        self.pulled_arms = []
        self.avg_rewards = []

    def choose_action(self, i_ph):
        # Activation rule
        not_covered = P.closed(lower=0, upper=1)
        for i in range(len(self.active_arms)):
            confidence_radius = self.confidence_radius(i_ph, i)
            # confidence_radius = math.sqrt((8 * i_ph)/1 + self.pulled_arms[i])
            confidence_interval = P.closed(self.active_arms[i] - confidence_radius, self.active_arms[i] + confidence_radius)
            not_covered = not_covered - confidence_interval
            if show:
                print(f"arm: {i}, i_ph: {i_ph}, pulled_arms: {self.pulled_arms[i]}")
                print(f"not_covered: {not_covered}, confidence_radius: {confidence_radius}")

        if not_covered != P.empty():
            rans = []
            height = 0
            heights = []
            for i in not_covered:
                rans.append(np.random.uniform(i.lower, i.upper))
                height += i.upper - i.lower
                heights.append(i.upper - i.lower)
            ran_n = np.random.uniform(0, height)
            j = 0
            for i in range(len(heights)):
                if j < ran_n < j + heights[i]:
                    ran = rans[i]
                j += heights[i]
            if test_pricing:
                ran = 0.1
            self.active_arms.append(ran)
            self.pulled_arms.append(0)
            self.avg_rewards.append(0)

        # Selection rule
        max_index = float('-inf')
        max_index_arm = None
        for i in range(len(self.active_arms)):
            confidence_radius = self.confidence_radius(i_ph, i)
            index = self.avg_rewards[i] + 2 * confidence_radius
            if index > max_index:
                max_index = index
                max_index_arm = i
        action = self.active_arms[max_index_arm]
        return action

    def learn(self, action, reward):
        # TODO: testing
        # reward = (reward + 200)/(270 + 200)
        if show:
            print(f"reward: {reward}")
        arm = self.active_arms.index(action)
        self.avg_rewards[arm] = (self.pulled_arms[arm] * self.avg_rewards[arm] + reward) / (self.pulled_arms[arm] + 1)
        self.pulled_arms[arm] += 1

    def confidence_radius(self, i_ph, i):
        return math.sqrt((8 * i_ph)/(1 + self.pulled_arms[i]))


class DPG:
    def __init__(self, agents, sigma_square=1, r_max=15, pricing_mechanism='LOO'):

        # self.beta = np.array([5, -3])
        self.sigma_square = sigma_square
        self.r_max = r_max
        self.agents = agents
        # shapley or LOO
        self.pricing_mechanism = pricing_mechanism

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
            x_s[i] = np.random.normal(self.agents[i].mu, self.agents[i].sigma_square ** .5)
            if test_pricing:
                x_s[0] = 0.055664941
                x_s[1] = 0.435856935

            if float(actions[i]) == 0:
                s_square_s[i] = self.agents[i].sigma_square
                x_tilde_s[i] = self.agents[i].mu
            else:
                s_square_s[i] = float(actions[i])
                x_tilde_s[i] = x_s[i] + np.random.normal(0, s_square_s[i] ** .5)

            if test_pricing:
                x_tilde_s[0] = 0.463488408
                x_tilde_s[1] = 0.282960863
            y_tilde_s[i] = x_tilde_s[i] * self.agents[i].beta
            # y_tilde = sum(x_tilde_s * self.beta)

        epsilon = np.random.normal(0, self.sigma_square ** .5)
        if test_pricing:
            epsilon = 0.047803865

        y = sum(x_s[i] * agents[i].beta for i in range(len(self.agents))) + epsilon
        # y = sum(x_s * self.beta) + epsilon
        # Y = X1 * self.beta1 + X2 * self.beta2 + epsilon
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

            if 0 < s_square_s[i] < self.agents[i].sigma_square:
                costs[i] = (self.agents[i].sigma_square / s_square_s[i]) + math.log10(s_square_s[i]) - (1 + math.log10(self.agents[i].sigma_square))
            elif self.agents[i].sigma_square < s_square_s[i]:
                costs[i] = 0

            # print(f"player {i}, costs: {costs[i]}")
            profits = prices - costs
            # print(f"player {i}, profit: {profits[i]}")
        return profits

    def zoom(self, rounds):

        i_ph = 0
        while i_ph < rounds:
            i_ph = i_ph + 1

            for t in range(1, 2 ** i_ph + 1):
                actions = []
                for agent in self.agents:
                    action = agent.choose_action(i_ph)
                    actions.append(action)

                rewards = self.step(actions)

                for j in range(len(agents)):
                    # print(f"player {j} action: {actions[j]} profit: {rewards[j]}")
                    self.agents[j].learn(actions[j], rewards[j])
                    # print(f"{type(action)}, {action}")
                    # print(action)
                self.episode_rewards.append(rewards)
                episode_reward = sum(rewards)
                self.episode_rewards_sum.append(episode_reward)

                if t % SHOW_EVERY == 0:
                    global show
                    show = True
                    # print(f"Episode: #{episode}")
                    print()
                    print(f"Episode {t}, {SHOW_EVERY} episode mean: {np.mean(self.episode_rewards_sum[-SHOW_EVERY:])}")
                    for j in range(len(agents)):
                        print(f"player: {j} action: {actions[j]} profit: {rewards[j]}")
                        print(f"active_arms: {agents[j].active_arms}")
                        print(f"pulled_arms: {agents[j].pulled_arms}")
                        print(f"avg_rewards: {agents[j].avg_rewards}")
                else:
                    show = False
        moving_avg_sum = np.convolve(self.episode_rewards_sum, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

        # moving_avg = []
        rewards_per_agent = list(zip(*self.episode_rewards))
        for j in range(len(self.agents)):
            moving_avg = np.convolve(rewards_per_agent[j], np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
            plt.plot([i for i in range(len(moving_avg))], moving_avg, label=j)
        plt.plot([i for i in range(len(moving_avg_sum))], moving_avg_sum, label='sum')

        plt.legend(loc='upper left')
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        plt.xlabel("episode #")
        plt.show()


class Arm:
    def __init__(self, value):
        value = value
        pulled = 0


if __name__ == "__main__":
    # training
    p0 = Agent(beta=5, sigma_square=0.5, mu=0)
    p1 = Agent(beta=-3, sigma_square=0.3, mu=0)
    p2 = Agent(beta=4, sigma_square=0.4, mu=0)
    p3 = Agent(beta=2, sigma_square=0.1, mu=0)
    agents = [p0, p1]

    dpg = DPG(agents, pricing_mechanism='shapley')
    print("training ...")
    dpg.zoom(12)
