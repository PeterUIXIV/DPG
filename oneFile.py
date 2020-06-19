import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math

style.use("ggplot")

SIZE = 1

HM_EPISODES = 100000
epsilon = 0.9  # randomness
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 10000  # how often to play through env visually.

start_q_table1 = None  # if we have a pickled Q table, we'll put the filename of it here.
start_q_table2 = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_1N = 1
PLAYER_2N = 2

d = {1: (255, 175, 0),  # blueish color
     2: (0, 0, 255)}  # green
action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
n_actions = len(action_space)

class DPG:

    def __init__(self):
        self.beta1 = 5
        self.beta2 = -3
        self.sigmaS = 1
        self.sigmaS1 = 0.5
        self.sigmaS2 = 0.3
        self.rmax = 15
        # Shapley or LOO
        self.pricing_mechanism = 'LOO'
        self.profit1 = 0
        self.profit2 = 0

    def action(self, choice1, choice2):

        global price1
        X1 = np.random.normal(0, self.sigmaS1 ** .5)
        X2 = np.random.normal(0, self.sigmaS2 ** .5)
        epsilon = np.random.normal(0, self.sigmaS ** .5)
        Y = X1 * self.beta1 + X2 * self.beta2 + epsilon

        if choice1 == '0':
            #TODO: X_tilde is set to the expected value of the corresponding predictor mu_j
            s_square1 = self.sigmaS1
            s_square2 = self.sigmaS2
        elif choice2 == '0':
            s_square1 = self.sigmaS1
            s_square2 = self.sigmaS2
        else:
            s_square1 = float(choice1)
            s_square2 = float(choice2)

        X1tilde = X1 + np.random.normal(0, s_square1 ** .5)
        X2tilde = X2 + np.random.normal(0, s_square2 ** .5)
        Ytilde_1 = X1tilde * self.beta1
        Ytilde_2 = X2tilde * self.beta2
        Ytilde = X1tilde * self.beta1 + X2tilde * self.beta2

        revenue0 = self.rmax - Y ** 2
        revenue1 = self.rmax - (Ytilde_1 - Y) ** 2
        revenue2 = self.rmax - (Ytilde_2 - Y) ** 2
        revenue12 = self.rmax - (Ytilde - Y) ** 2

        if self.pricing_mechanism == 'Shapley':
            price1 = 0.5 * (revenue12 - revenue2) + 0.5 * (revenue1 - revenue0)
            price2 = 0.5 * (revenue12 - revenue1) + 0.5 * (revenue2 - revenue0)
        elif self.pricing_mechanism == 'LOO':
            price1 = revenue12 - revenue2
            price2 = revenue12 - revenue1

        # print(f"choice1: {choice1}, choice2: {choice2}")
        # print(f"s_square1: {s_square1}, s_square2 {s_square2}")
        costs1 = (self.sigmaS1/s_square1) + math.log10(s_square1) - (1 + math.log10(self.sigmaS1))
        costs2 = (self.sigmaS2/s_square2) + math.log10(s_square2) - (1 + math.log10(self.sigmaS2))

        self.profit1 = price1 - costs1
        self.profit2 = price2 - costs2


if start_q_table1 is None:
    q_table1 = {}
    for i in range(SIZE):
        q_table1[i] = [np.random.uniform(-15, 15) for i in range(10)]
else:
    with open(start_q_table1, "rb") as f:
        q_table1 = pickle.load(f)

if start_q_table2 is None:
    q_table2 = {}
    for i in range(SIZE):
        q_table2[i] = [np.random.uniform(-15, 15) for i in range(10)]
else:
    with open(start_q_table2, "rb") as f:
        q_table2 = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    dpg = DPG()
    #new parameter
    episode_reward = 0


    if np.random.uniform() > epsilon:
        #only one observation/state
        action1Id = np.argmax(q_table1[0])
        action1 = action_space[action1Id]
        action2Id = np.argmax(q_table2[0])
        action2 = action_space[action2Id]
    else:
        action1Id = np.random.randint(0, 10)
        action1 = action_space[action1Id]
        action2Id = np.random.randint(0, 10)
        action2 = action_space[action2Id]

    dpg.action(action1, action2)

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
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
    #new_q1 = (1 - LEARNING_RATE) * current_q1 + LEARNING_RATE * (dpg.price1 + DISCOUNT * max_future_q)

    episode_reward += dpg.profit1 + dpg.profit2
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    #print(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

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
