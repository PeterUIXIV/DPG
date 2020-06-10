import pandas as pd
import numpy as np

class QLearningTable:
    def __init__(self, actions, learning_rate=0.9, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.foat64)


