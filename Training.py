import DPG
import numpy as np
import load_dataset
import Agents
import matplotlib.pyplot as plt
import math
from py_expression_eval import Parser

# training
rounds = 10000

# setups: 'eps_greedy', 'stretch', 'logistic', 'real', 'bounded'
setup = "eps_greedy"
# cost functions: 'log', 'exp'
cost_function = 'exp'
# data set parameter
DATA_PATH = r'D:\workspaces\datasets\USCensus1990\USCensus1990.data.txt'
predictor = ['iMobillim', 'iWork89']
target = 'iMilitary'
with_data_set = False

# agent hyperparameter
# rho = 0.998
# v1 = 1.5
rho = 0.5
v1 = 1
# agent1 = Agents.HierarchicalOptimisticOptimization([0, 1], v1=v1, rho=rho, setup=setup)
# agent2 = Agents.HierarchicalOptimisticOptimization([0, 1], v1=v1, rho=rho, setup=setup)
# agent1 = Agents.Zooming([0, 1])
# agent2 = Agents.Zooming([0, 1])
# agent1 = Agents.Random([0, 1])
# agent2 = Agents.Random([0, 1])
discrete_actions = [[0.01, 0.07, 0.13, 0.19, 0.25, 0.31, 0.37, 0.43, 0.49],
                    [0.01, 0.045, 0.08, 0.115, 0.15, 0.185, 0.22, 0.255, 0.29]]
agent1 = Agents.EpsilonGreedy(discrete_actions[0], eps_greedy=0.999, eps_decay=0.9996)
agent2 = Agents.EpsilonGreedy(discrete_actions[1], eps_greedy=0.999, eps_decay=0.9996)

# DPG parameter
logistic_growth = 0.2
betas = [5, -3]
# betas = [2, -3]
sigmas_j_square = [.5, .3]
# sigmas_j_square = [.7, .3]
# 'loo' or 'shapley'
pricing_mechanism = 'loo'
r_max = 15
sigma_square = 1

'''
def cost_function(sigma_square, s_square):
    return (sigma_square / s_square) + math.log10(s_square) - (1 + math.log10(sigma_square))
'''
# alt_cost_function = "sigma_square * exp(-5 * (s_square - sigma_square)) -sigma_square"
# new_cost_function = 'sigma_square / (s_square + 0.1) + log((s_square + 0.1), 10) - (1 + log(sigma_square, 10)) - log((sigma_square + 0.1)/sigma_square) + 0.1/(sigma_square + 0.1)'
# cost_function = '(sigma_square / s_square) + log(s_square, 10) - (1 + log(sigma_square, 10))'
# alt2_cost_function = "sigma_square * exp(-5 * ((s_square - 0.2) - sigma_square)) -sigma_square * exp(-5*(-0.2))"
parser = Parser()
if cost_function == 'log':
    cost_log = "(sigma_square / s_square) + log(s_square, 10) - (1 + log(sigma_square, 10))"
    cost_function = parser.parse(cost_log)
    optimal_rewards = [6.991, 0.658]
elif cost_function == 'exp':
    cost_exp = "sigma_square * exp(-5 * (s_square - 0.2 - sigma_square)) -sigma_square * exp(-5*(-0.2))"
    cost_function = parser.parse(cost_exp)
    optimal_rewards = [2.8721, 0.4407]
else:
    print("no cost function!!!")
    cost_exp = "sigma_square * exp(-5 * ((s_square - 0.2) - sigma_square)) -sigma_square * exp(-5*(-0.2))"
    cost_function = parser.parse(cost_exp)
    optimal_rewards = [2.8721, 0.4407]


if __name__ == "__main__":
    SHOW_EVERY = int(rounds / 5)

    if with_data_set:
        data_set = load_dataset.DataSet(path=DATA_PATH, predictor=predictor, target=target)
        betas = data_set.coef
        sigmas_j_square = data_set.var
    else:
        data_set = None

    data_provider = [DPG.DataProvider(mu=0, sigma_square=sigmas_j_square[0], cost_function=cost_function, agent=agent1,
                                      setup=setup),
                     DPG.DataProvider(mu=0, sigma_square=sigmas_j_square[1], cost_function=cost_function, agent=agent2,
                                      setup=setup)]

    print(betas)
    dpg = DPG.DataProvisionGame(data_provider,
                                DPG.AnalyticsServiceProvider(np.array(betas)),
                                DPG.DataConsumer(pricing_mechanism=pricing_mechanism, r_max=r_max), data_set=data_set,
                                sigma_square=sigma_square, logistic_growth=logistic_growth)

    # optimal_rewards = [9.403, 1.4329]
    # optimal_rewards = [6.991, 0.658]
    # optimal_rewards = [1.454, 1.4329]
    # optimal_rewards = [2.8721, 0.4407]

    for episode in range(1, rounds + 1):
        dpg.step(episode)
        if episode % SHOW_EVERY == 0:
            print(f"episode {episode} of {rounds}")
            for i in range(len(dpg.data_provider_list)):
                rewards = dpg.data_provider_list[i].rewards
                length = len(dpg.data_provider_list[i].rewards)
                print(f"Agent {i+1}")
                print(f"mean reward: {sum(rewards)/length}")
                print(f"mean regret: {sum(optimal_rewards[i]-np.asarray(rewards))/length}")
                print(f"mean reward over last {SHOW_EVERY}: {sum(rewards[-SHOW_EVERY:])/SHOW_EVERY}")
                print(f"mean regret over last {SHOW_EVERY}: {sum(optimal_rewards[i] - np.asarray(rewards)[-SHOW_EVERY:]) / SHOW_EVERY}")
                print(f"accumulated reward: {sum(rewards)}")
                print(f"accumulated regret: {sum(optimal_rewards[i]-np.asarray(rewards))}")
                dpg.data_provider_list[i].agent.show_learning()

    rewards = []
    regrets = []
    for i in range(len(dpg.data_provider_list)):
        rewards.append(np.asarray(dpg.data_provider_list[i].rewards))
        regrets.append(optimal_rewards[i]-np.asarray(dpg.data_provider_list[i].rewards))
    # overall_rewards = sum(rewards)

    # overall_moving_avg = np.convolve(overall_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

    for i in range(len(rewards)):
        rewards_moving_avg = np.convolve(rewards[i], np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        regrets_moving_avg = np.convolve(regrets[i], np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        plt.plot([j for j in range(len(rewards_moving_avg))], rewards_moving_avg, label=f"rewards {i+1}")
        plt.plot([j for j in range(len(regrets_moving_avg))], regrets_moving_avg, label=f"regrets {i+1}")

    # plt.plot([i for i in range(len(overall_moving_avg))], overall_moving_avg, label='sum of rewards')

    plt.legend(loc='upper left')
    plt.ylabel(f"Rewards and regrets {SHOW_EVERY}ma")
    plt.xlabel(f"time step")
    plt.show()
