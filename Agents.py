import Node
import random
import math
import numpy as np
import pandas as pd
import portion as P


class HierarchicalOptimisticOptimization:
    def __init__(self, action_space, v1, rho):
        # value functions: action_value_method, q-learning, UCB
        self.v1 = v1
        self.rho = rho

        root = Node.Node(action_space[0], action_space[1], 0, 1, 0)
        root.active = True
        root.left = Node.Node(root.lower, (root.lower + root.higher) / 2, root.h + 1, root.i * 2 - 1, float("inf"))
        root.right = Node.Node((root.lower + root.higher) / 2, root.higher, root.h + 1, root.i * 2, float("inf"))
        self.root = root
        self.highest_b_leaf = None

        self.row_list = []
        self.lower_limit = float("-inf")
        self.upper_limit = float("inf")
        self.below_limit = 0

    def choose_action(self, episode):
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

    def learn(self, reward, n):
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
            node.u = node.mean + math.sqrt((2 * math.log(n)) / node.count) + self.v1 * math.pow(self.rho, node.h)

        highest_b_leaf.left = Node.Node(highest_b_leaf.lower, (highest_b_leaf.lower + highest_b_leaf.higher) / 2,
                                        highest_b_leaf.h + 1, highest_b_leaf.i * 2 - 1, float("inf"))
        highest_b_leaf.right = Node.Node((highest_b_leaf.lower + highest_b_leaf.higher) / 2, highest_b_leaf.higher,
                                         highest_b_leaf.h + 1, highest_b_leaf.i * 2, float("inf"))

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
            parent, leaf = tree_copy.first_leaf_and_parent()
            node = self.root.find(leaf)
            node.b = min(node.u, max(node.left.b, node.right.b))
            parent.remove(leaf)

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

    def show_learning(self):
        self.root.print_tree_depth(4)


class Arm:
    def __init__(self, index, value, norm_value):
        self.index = index
        self.value = value
        self.norm_value = norm_value
        self.pulled = 0
        self.avg_learning_reward = 0
        self.avg_reward = 0


class Zooming:
    def __init__(self, action_space):
        # value functions: action_value_method, q-learning, UCB
        # TODO: each DP saves his rewards and does not consider highest and lowest 1% of rewards
        self.action_space = action_space

        self.active_arms = []
        self.pulled_arms = []
        self.avg_rewards = []
        self.max_profit = 0
        self.min_profit = 0
        self.high_count = 0
        self.low_count = 0

        self.current_arm = None

    def choose_action(self, episode):
        i_ph = int(math.log2(episode)) + 1
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
            new_arm = Arm(len(self.active_arms), ran * (self.action_space[1] - self.action_space[0]) + self.action_space[0], ran)
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

    def learn(self, reward, i):
        arm = self.current_arm
        arm.avg_reward = (arm.pulled * arm.avg_reward + reward) / (arm.pulled + 1)
        # action = (action - lower)/(self.sigma_square - lower)
        # action = self.current_action
        if reward > self.max_profit:
            self.max_profit = reward
        elif reward < self.min_profit:
            self.min_profit = reward

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
            reward = (reward - low)/(high - low)

        arm.avg_learning_reward = (arm.pulled * arm.avg_learning_reward + reward) / (arm.pulled + 1)
        arm.pulled += 1

    def show_learning(self):
        pass


def calc_confidence_radius(i_ph, arm: Arm):
    return math.sqrt((8 * i_ph)/(1 + arm.pulled))


class EpsilonGreedy:
    def __init__(self, action_space, eps_greedy=0.95, eps_decay=0.9995,
                 mu=0):
        self.action_space = action_space
        # gamma, discount, reward decay
        self.eps_decay = eps_decay
        # epsilon
        self.eps_greedy = eps_greedy
        # Q-table
        self.q_table = pd.DataFrame(0, index=[0], columns=action_space)
        # value functions: action_value_method, q-learning, UCB

    def choose_action(self, rounds):
        if np.random.uniform() > self.eps_greedy:
            # only one observation/state
            # exploitation
            state_action = self.q_table.loc[0, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # exploration
            action = np.random.choice(self.action_space)
        return action

    def learn(self, action, reward, episode):
        # update q_table
        current_q = self.q_table.loc[0, action]
        new_q = current_q + (1/(episode + 1)) * (reward - current_q)
        self.q_table.loc[0, action] = new_q

        self.eps_greedy *= self.eps_decay
