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
import random
# from anytree import Node, RenderTree, PreOrderIter
import Node
style.use("ggplot")

# SHOW_EVERY = 1000  # how often to play through env visually.

lower = 0.0


class Agent:
    def __init__(self, beta, sigma_square, mu=0):
        # value functions: action_value_method, q-learning, UCB
        # TODO: each DP saves his rewards and does not consider highest and lowest 1% of rewards
        self.mu = mu
        self.beta = beta
        self.sigma_square = sigma_square

        self.actions = []

        root = Node.Node(lower, sigma_square, 0, 1, 0)
        root.active = True
        root.left = Node.Node(root.lower, (root.lower + root.higher)/2, root.h+1, root.i*2-1, float("inf"))
        root.right = Node.Node((root.lower + root.higher)/2, root.higher, root.h+1, root.i*2, float("inf"))
        self.root = root
        self.highest_b_leaf = None

        self.row_list = []
        self.lower_limit = float("-inf")
        self.upper_limit = float("inf")
        self.below_limit = 0

    def choose_action(self):
        node = self.root
        while node.left or node.right:
            if node.left.b > node.right.b:
                node = node.left
            elif node.left.b < node.right.b:
                node = node.right
            else:
                node = random.choice([node.left, node.right])

        self.highest_b_leaf = node
        # choose arm X in P_H,I and play it

        # Calculating interval from binary tree

        action = np.random.uniform(node.lower, node.higher)

        return action

    def learn(self, reward, n, v1, rho):
        highest_b_leaf = self.highest_b_leaf

        # remove outliers
        self.row_list.append({'node': highest_b_leaf, 'value': reward})

        # look for outliers
        if n >= 100:
            if n == 100 or math.log(n, 2) % 2 == 0:
                df = pd.DataFrame(self.row_list, columns=['node', 'value'])
                q_1, q_3 = df.value.quantile([0.25, 0.75])
                iqr = q_3 - q_1
                self.lower_limit = q_1 - 1.5 * iqr
                self.upper_limit = q_3 + 1.5 * iqr
                print(f"Round {n} lower limit: {self.lower_limit}, upper limit: {self.upper_limit}")

            if reward < self.lower_limit:
                self.below_limit += 1
                # print(f"Skipped ({highest_b_leaf.lower}, {highest_b_leaf.higher}), {reward}")
                reward = self.lower_limit

        # extend the tree
        highest_b_leaf.active = True

        reward = reward / 15

        # Update the statistics node.count and node.mean
        for node in self.root.path(highest_b_leaf):
            node.count += 1
            if node.mean is None:
                node.mean = reward
            else:
                node.mean = (1 - 1 / node.count) * node.mean + reward / node.count

        # Update the statistics node.u stored in the tree
        for node in self.root.pre_order_traversal():
            node.u = node.mean + math.sqrt((2 * math.log(n)) / node.count) + v1 * math.pow(rho, node.h)

        highest_b_leaf.left = Node.Node(highest_b_leaf.lower, (highest_b_leaf.lower + highest_b_leaf.higher)/2,
                                        highest_b_leaf.h+1, highest_b_leaf.i*2-1, float("inf"))
        highest_b_leaf.right = Node.Node((highest_b_leaf.lower + highest_b_leaf.higher)/2, highest_b_leaf.higher,
                                         highest_b_leaf.h+1, highest_b_leaf.i*2, float("inf"))

        tree_copy = self.root.duplicate_tree()
        while True:
            if tree_copy.left:
                if not tree_copy.left.active:
                    break
            elif tree_copy.right:
                if not tree_copy.right.active:
                    break
            else:
                break
            try:
                parent, leaf = tree_copy.first_leaf_and_parent()
                node = self.root.find(leaf)
                node.b = min(node.u, max(node.left.b, node.right.b))
                parent.remove(leaf)
            except AttributeError as e:
                print(f"parent ({parent.lower}, {parent.higher}), h: {parent.h}, i: {parent.i}, b-value: {parent.b}")
                print(f"leaf ({leaf.lower}, {leaf.higher}), h: {leaf.h}, i: {leaf.i}, b-value: {leaf.b}")
                print(f"node ({node.lower}, {node.higher}), h: {node.h}, i: {node.i}, b-value: {node.b}")

        # remove outliers
        if n == 100:
            df = pd.DataFrame(self.row_list, columns=['node', 'value'])
            q_1, q_3 = df.value.quantile([0.25, 0.75])
            iqr = q_3 - q_1
            lower_limit = q_1 - 1.5 * iqr
            for index, row in df.iterrows():
                if row['value'] < lower_limit:
                    print(f"Removed ({row['node'].lower}, {row['node'].higher}), reward {row['value']}")
                    for node in reversed(self.root.path(row['node'])):
                        if node.count == 1:
                            self.root.find_and_remove(node)
                        else:
                            node.mean = ((node.mean * node.count) - row['value']) / (node.count - 1)
                            node.count -= 1



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

    def play(self, v1, rho, n):

        for i in range(1, n + 1):
            if i % SHOW_EVERY == 0:
                globals()['show'] = True
                print()
                print(f"Episode {i} of {n}")
            else:
                globals()['show'] = False
            actions = []
            for agent in self.agents:
                action = agent.choose_action()
                actions.append(action)

            # receive corresponding reward Y
            rewards = self.step(actions)

            for j in range(len(agents)):
                # print(f"player {j} action: {actions[j]} profit: {rewards[j]}")
                if debug:
                    print(f"agent: {j}")
                self.agents[j].learn(rewards[j], i, v1, rho)
                # print(f"{type(action)}, {action}")
                # print(action)
            self.episode_rewards.append(rewards)
            episode_reward = sum(rewards)
            self.episode_rewards_sum.append(episode_reward)

            if show:
                # avg_rewards_per_agent = []
                # for list_of_rewards in rewards_per_agent:
                #     avg_rewards_per_agent.append(statistics.mean(list_of_rewards))
                print()
                print(f"{SHOW_EVERY} episode mean: {np.mean(self.episode_rewards_sum[-SHOW_EVERY:])} v1*rho^h: {v1*(rho**4)}")
                for j in range(len(agents)):
                    print()
                    print(f"player: {j} action: {actions[j]} profit: {rewards[j]}")
                    print(f"# Below limit rewards: {agents[j].below_limit}")
                    agents[j].root.print_tree_depth(4)

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

    rounds = 10000
    SHOW_EVERY = int(rounds / 5)

    show = False
    # training
    p0 = Agent(beta=5, sigma_square=0.5, mu=0)
    p1 = Agent(beta=-3, sigma_square=0.3, mu=0)
    p2 = Agent(beta=4, sigma_square=0.4, mu=0)
    p3 = Agent(beta=2, sigma_square=0.1, mu=0)
    agents = [p0, p1]

    # hyperparameter
    rho = 0.998
    v1 = 1.5

    dpg = DPG(agents, pricing_mechanism='LOO')
    print("training ...")
    dpg.play(v1=v1, rho=rho, n=rounds)
