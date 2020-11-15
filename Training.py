import DPG
import numpy as np
import load_dataset
import Agents
import matplotlib.pyplot as plt
from py_expression_eval import Parser

# training
rounds = 1000

# data set parameter
DATA_PATH = r'D:\workspaces\datasets\USCensus1990\USCensus1990.data.txt'
predictor = ['iMobillim', 'iWork89']
target = 'iMilitary'
with_data_set = True

# agent hyperparameter
rho = 0.998
v1 = 1.5
# agent1 = Agents.HierarchicalOptimisticOptimization([0, 0.5], v1=v1, rho=rho)
# agent2 = Agents.HierarchicalOptimisticOptimization([0, 0.3], v1=v1, rho=rho)
agent1 = Agents.Zooming([0.01, 0.5])
agent2 = Agents.Zooming([0.01, 0.3])

# DPG parameter
betas = [5, -3]
pricing_mechanism = 'loo'
r_max = 15
sigma_square = 1

'''
def cost_function(sigma_square, s_square):
    return (sigma_square / s_square) + math.log10(s_square) - (1 + math.log10(sigma_square))
'''
parser = Parser()
# cost_function = '(sigma_square / s_square) + math.log10(s_square) - (1 + math.log10(sigma_square))'
cost_function = '(sigma_square / s_square) + log(s_square, 10) - (1 + log(sigma_square, 10))'
cost_function = parser.parse(cost_function)
data_provider = [DPG.DataProvider(mu=0, sigma_square=0.5, cost_function=cost_function, agent=agent1),
                 DPG.DataProvider(mu=0, sigma_square=0.3, cost_function=cost_function, agent=agent2)]


if __name__ == "__main__":
    SHOW_EVERY = int(rounds / 5)

    if with_data_set:
        data_set = load_dataset.DataSet(path=DATA_PATH, predictor=predictor, target=target)
        betas = data_set.coef
    else:
        data_set = None

    dpg = DPG.DataProvisionGame(data_provider,
                                DPG.AnalyticsServiceProvider(np.array(betas)),
                                DPG.DataConsumer(pricing_mechanism=pricing_mechanism, r_max=r_max), data_set=data_set,
                                sigma_square=sigma_square)

    for episode in range(1, rounds + 1):
        dpg.step(episode)
        if episode % SHOW_EVERY == 0:
            print(f"episode {episode}")
            for i in range(len(dpg.data_provider_list)):
                print(f"Agent {i+1}")
                print(f"Avg rewards: {sum(dpg.data_provider_list[i].rewards)/len(dpg.data_provider_list[i].rewards)}")
                print(f"accumulated rewards: {sum(dpg.data_provider_list[i].rewards)}")
                dpg.data_provider_list[i].agent.show_learning()

    rewards = []
    for i in range(len(dpg.data_provider_list)):
        rewards.append(np.asarray(dpg.data_provider_list[i].rewards))
    overall_rewards = sum(rewards)

    overall_moving_avg = np.convolve(overall_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

    for i in range(len(rewards)):
        moving_avg = np.convolve(rewards[i], np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
        plt.plot([j for j in range(len(moving_avg))], moving_avg, label=i)
    plt.plot([i for i in range(len(overall_moving_avg))], overall_moving_avg, label='sum')

    plt.legend(loc='upper left')
    plt.ylabel(f"Rewards {SHOW_EVERY}ma")
    plt.xlabel(f"episode #")
    plt.show()
