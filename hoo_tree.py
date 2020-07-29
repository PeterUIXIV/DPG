import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import itertools
import pandas as pd
import portion as P
import statistics
from random import random, randint
from anytree import Node, RenderTree, PreOrderIter

style.use("ggplot")

# SHOW_EVERY = 1000  # how often to play through env visually.

lower = 0.01


# action_space = ['0', '0.025', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8', '1']


class Arm:
    def __init__(self, index, value, norm_value):
        self.index = index
        self.value = value
        self.norm_value = norm_value
        self.pulled = 0
        self.avg_learning_reward = 0
        self.avg_reward = 0


class Agent:
    def __init__(self, beta, sigma_square, mu=0):
        # value functions: action_value_method, q-learning, UCB
        # TODO: each DP saves his rewards and does not consider highest and lowest 1% of rewards
        self.mu = mu
        self.beta = beta
        self.sigma_square = sigma_square

        self.active_arms = []
        self.pulled_arms = []
        self.avg_rewards = []
        self.max_profit = 0
        self.min_profit = 0
        self.high_count = 0
        self.low_count = 0

        self.current_arm = 0

    def choose_action(self, i_ph):
        # Activation rule
        not_covered = P.closed(lower=0, upper=1)
        # not_covered = P.closed(lower=0.01, upper=self.sigma_square)
        # scale = not_covered.upper - not_covered.lower
        for arm in self.active_arms:
            # confidence_radius = scale * self.confidence_radius(i_ph, i)
            confidence_radius = calc_confidence_radius(i_ph, arm)
            confidence_interval = P.closed(arm.norm_value - confidence_radius,
                                           arm.norm_value + confidence_radius)
            not_covered = not_covered - confidence_interval
            if debug:
                print(f"arm: {arm.index}, value: {arm.value}, i_ph: {i_ph}, pulled_arms: {arm.pulled}")
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
            ran = 0
            for i in range(len(heights)):
                if j < ran_n < j + heights[i]:
                    ran = rans[i]
                j += heights[i]
            if debug_pricing:
                ran = 0.1
            new_arm = Arm(len(self.active_arms), ran * (self.sigma_square - lower) + lower, ran)
            self.active_arms.append(new_arm)
            # self.pulled_arms.append(0)
            # self.avg_rewards.append(0)

        # Selection rule
        max_index = float('-inf')
        max_index_arm = None
        for arm in self.active_arms:
            confidence_radius = calc_confidence_radius(i_ph, arm)
            index = arm.avg_learning_reward + 2 * confidence_radius
            if index > max_index:
                max_index = index
                max_index_arm = arm
        action = max_index_arm.value
        self.current_arm = max_index_arm
        # self.current_action = action
        # action = action * (self.sigma_square - lower) + lower
        return action

    def learn(self, action, reward):
        arm = self.current_arm
        arm.avg_reward = (arm.pulled * arm.avg_reward + reward) / (arm.pulled + 1)
        # action = (action - lower)/(self.sigma_square - lower)
        # action = self.current_action
        if reward > self.max_profit:
            self.max_profit = reward
        elif reward < self.min_profit:
            self.min_profit = reward
        if debug:
            print(f"action: {action}, reward: {reward}")
        # TODO: testing

        # normalize reward to [0, 1]
        high = 100
        low = -75
        if reward >= high:
            reward = 1
            self.high_count += 1
        elif reward <= low:
            reward = 0
            self.low_count += 1
        else:
            reward = (reward - low) / (high - low)

        # reward = reward / 20
        if debug:
            print(f"high_count: {self.high_count}, low_count: {self.low_count}")

        arm.avg_learning_reward = (arm.pulled * arm.avg_learning_reward + reward) / (arm.pulled + 1)
        arm.pulled += 1


def calc_confidence_radius(i_ph, arm: Arm):
    return math.sqrt((8 * i_ph) / (1 + arm.pulled))


def get_leaf(tree):
    for node in tree:
        if (node[0] + 1, node[0] * 2 - 1) not in tree and (node[1] + 1, node[1] * 2) not in tree:
            return node


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

    '''
    def max_radius(self, z=4.891638, agents):
        x_s = np.array(z*agents.beta)
    '''

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
            if debug_pricing:
                x_s[0] = -0.473797659
                x_s[1] = 0.121024593

            if float(actions[i]) == 0:
                s_square_s[i] = self.agents[i].sigma_square
                x_tilde_s[i] = self.agents[i].mu
            else:
                s_square_s[i] = float(actions[i])
                x_tilde_s[i] = x_s[i] + np.random.normal(0, s_square_s[i] ** .5)

            if debug_pricing:
                x_tilde_s[0] = -0.426597148
                x_tilde_s[1] = 0.650116791
            y_tilde_s[i] = x_tilde_s[i] * self.agents[i].beta
            # y_tilde = sum(x_tilde_s * self.beta)

        epsilon = np.random.normal(0, self.sigma_square ** .5)
        if debug_pricing:
            epsilon = 0.10286918

        y = sum(x_s[i] * agents[i].beta for i in range(len(self.agents))) + epsilon

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
                costs[i] = (self.agents[i].sigma_square / s_square_s[i]) + math.log10(s_square_s[i]) - (
                        1 + math.log10(self.agents[i].sigma_square))
            elif self.agents[i].sigma_square < s_square_s[i]:
                costs[i] = 0

            # print(f"player {i}, costs: {costs[i]}")
            profits = prices - costs
            # print(f"player {i}, profit: {profits[i]}")
        return profits

    def hoo(self, v1, rho, subsets, n, rounds):
        node_counter = {}
        mean_values = {}
        actions = []
        u_values = {}

        root = Node(P.open(0, 1), parent=None)
        b_values = {(1, 2): float("inf"), (2, 2): float("inf")}
        tree = [(0, 1)]
        path = []
        for j in range(n):
            (h, i) = (0, 1)
            path.append((h, i))
            node = Node((h, i), )

            while (h, i) in tree:
                if b_values[(h + 1, 2 * i - 1)] > b_values[(h + 1, 2 * i - 1)]:
                    (h, i) = (h + 1, 2 * i - 1)
                elif b_values[(h + 1, 2 * i - 1)] < b_values[h + 1, 2 * i]:
                    (h, i) = (h + 1, 2 * i)
                else:
                    (h, i) = (h + 1, 2 * i - random.ranint(0, 1))
                path.append((h, i))
            (H, I) = (h, i)
            node_counter[h, i] = 0
            # TODO: choose arm X inP_H,I and play it
            # TODO: receive corresponding reward Y
            reward = np.random.uniform(-10, 10)

            tree.append((H, I))
            for (h, i) in path:
                node_counter[(h, i)] = node_counter[(h, i)] + 1
                mean_values[(h, i)] = (1 - 1 / node_counter[(h, i)]) * mean_values[(h, i)] + reward / node_counter[(h, i)]
            for (h, i) in tree:
                u_values[(h, i)] = mean_values[(h, i)] + math.sqrt((2 * math.log(j))/node_counter[(h, i)]) + v1 * math.pow(rho, h)
            b_values[(H + 1, 2 * I - 1)] = float("inf")
            b_values[(H+1, 2*I)] = float("inf")
            tree_c = tree
            while tree_c != [(0, 1)]:
                (h, i) = get_leaf(tree_c)
                b_values[(h, i)] = min(u_values[(h, i)], max(b_values[(h + 1, i * 2 - 1)], b_values[(h + 1, i * 2)]))
                tree_c.remove((h, i))

        k = 0
        i_ph = 0
        while i_ph < rounds:
            i_ph = i_ph + 1

            for t in range(1, 2 ** i_ph + 1):
                k += 1
                if k % SHOW_EVERY == 0:
                    globals()['show'] = True
                    print()
                    print(f"Phase {i_ph} of {rounds}, Episode {t} of {2 ** i_ph}")
                else:
                    globals()['show'] = False
                actions = []
                for agent in self.agents:
                    action = agent.choose_action(i_ph)
                    actions.append(action)

                rewards = self.step(actions)

                for j in range(len(agents)):
                    # print(f"player {j} action: {actions[j]} profit: {rewards[j]}")
                    if debug:
                        print(f"agent: {j}")
                    self.agents[j].learn(actions[j], rewards[j])
                    # print(f"{type(action)}, {action}")
                    # print(action)
                self.episode_rewards.append(rewards)
                episode_reward = sum(rewards)
                self.episode_rewards_sum.append(episode_reward)

                if show:
                    rewards_per_agent = list(zip(*self.episode_rewards))
                    # avg_rewards_per_agent = []
                    # for list_of_rewards in rewards_per_agent:
                    #     avg_rewards_per_agent.append(statistics.mean(list_of_rewards))
                    print(f"{SHOW_EVERY} episode mean: {np.mean(self.episode_rewards_sum[-SHOW_EVERY:])}")
                    for j in range(len(agents)):
                        print(f"player: {j} action: {actions[j]} profit: {rewards[j]}")
                        print(f"max_profit: {agents[j].max_profit}, min_profit: {agents[j].min_profit}")
                        for arm in agents[j].active_arms:
                            print(f"arm: {arm.index}, value: {arm.value}, pulled: {arm.pulled}, "
                                  f"avg_learning_reward: {arm.avg_learning_reward}, avg_reward: {arm.avg_reward}")

                        # print(f"avg_real_rewards: {avg_rewards_per_agent[j]}")
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


if __name__ == "__main__":
    debug = False
    debug_pricing = False

    rounds = 20
    SHOW_EVERY = int((2 ** rounds) / 10)

    show = False
    # training
    p0 = Agent(beta=5, sigma_square=0.5, mu=0)
    p1 = Agent(beta=-3, sigma_square=0.3, mu=0)
    p2 = Agent(beta=4, sigma_square=0.4, mu=0)
    p3 = Agent(beta=2, sigma_square=0.1, mu=0)
    agents = [p0, p1]

    dpg = DPG(agents, pricing_mechanism='LOO')
    print("training ...")
    dpg.hoo(rounds)
