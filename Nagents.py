import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
import itertools

style.use("ggplot")

SIZE = 1

HM_EPISODES = 100000
eps = 0.9  # randomness
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000  # how often to play through env visually.

start_q_table1 = None  # if we have a pickled Q table, we'll put the filename of it here.
start_q_table2 = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
n_actions = len(action_space)


class DPG:
    def __init__(self):
        self.mu = np.array([0, 0])
        self.sigma_square_s = np.array([0.5, 0.3])
        self.beta = np.array([5, -3])
        self.sigma_square = 1
        self.r_max = 15
        # Shapley or LOO
        self.pricing_mechanism = 'LOO'

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
                    p += self.v(perm[:i + 1], y_tilde_s, y) - self.v(perm[:i], y_tilde_s, y)
        return p / math.factorial(n)

    def loo(self, y_tilde_s, y, j, n):
        x = list(range(n))
        x.remove(j)
        p = self.v(range(n), y_tilde_s, y) - self.v(x, y_tilde_s, y)
        return p

    def action(self, choices):
        n = len(choices)
        s_square_s = np.zeros(n)
        x_s = np.zeros(n)
        x_tilde_s = np.zeros(n)
        y_tilde_s = np.zeros(n)
        i: int
        for i in range(n):
            x_s[i] = np.random.normal(self.mu[i], self.sigma_square_s[i])

            if choices[i] == '0':
                s_square_s[i] = self.sigma_square_s[i]
                x_tilde_s[i] = self.mu[i]
            else:
                s_square_s[i] = float(choices[i])
                x_tilde_s[i] = x_s[i] + np.random.normal(0, s_square_s[i] ** .5)

            y_tilde_s[i] = x_tilde_s[i] * self.beta[i]
            y_tilde = sum(x_tilde_s * self.beta)

        epsilon = np.random.normal(0, self.sigma_square ** .5)

        y = sum(x_s * self.beta) + epsilon
        # Y = X1 * self.beta1 + X2 * self.beta2 + epsilon

        prices = np.zeros(n)
        costs = np.zeros(n)
        profits = np.zeros(n)
        for i in range(n):
            if self.pricing_mechanism == 'shapley':
                prices[i] = self.shapley(y_tilde_s, y, i, n)
            elif self.pricing_mechanism == 'LOO':
                prices[i] = self.loo(y_tilde_s, y, i, n)
            costs[i] = (self.sigma_square_s[i] / s_square_s[i]) + math.log10(s_square_s[i]) - \
                       (1 + math.log10(self.sigma_square_s[i]))
            profits = prices - costs
        return profits
        # TODO: is this correct?


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

episode_rewards = []

for episode in range(HM_EPISODES):
    dpg = DPG()
    # new parameter
    episode_reward = 0

    if np.random.uniform() > eps:
        # only one observation/state
        action1Id = int(np.argmax(q_table1[0]))
        action1 = action_space[action1Id]
        action2Id = int(np.argmax(q_table2[0]))
        action2 = action_space[action2Id]
    else:
        action1Id = np.random.randint(0, 10)
        action1 = action_space[action1Id]
        action2Id = np.random.randint(0, 10)
        action2 = action_space[action2Id]

    dpg.action(action1, action2)

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {eps}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        print(f"player1 action: {action1} profit2: {dpg.profit1}")
        print(f"player2 action: {action2} profit2 {dpg.profit2}")
        print(q_table1)

        show = True
    else:
        show = False

    max_future_q1 = np.max(q_table1[0])
    max_future_q2 = np.max(q_table1[0])
    current_q1 = q_table1[0][action1Id]
    current_q2 = q_table2[0][action2Id]

    new_q1 = current_q1 + LEARNING_RATE * ((dpg.profit1 + DISCOUNT * max_future_q1) - current_q1)
    new_q2 = current_q2 + LEARNING_RATE * ((dpg.profit2 + DISCOUNT * max_future_q2) - current_q2)

    q_table1[0][action1Id] = new_q1
    q_table2[0][action2Id] = new_q2
    # new_q1 = (1 - LEARNING_RATE) * current_q1 + LEARNING_RATE * (dpg.price1 + DISCOUNT * max_future_q)

    episode_reward += dpg.profit1 + dpg.profit2
    episode_rewards.append(episode_reward)
    eps *= EPS_DECAY
    # print(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
# plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
# plt.ylabel(f"Reward")
# plt.xlabel(f"episode #")
# plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table1, f)
