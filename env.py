import numpy as np

class Environment():
    def __init__(self):
        self.action_space = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
        self.n_actions = len(self.action_space)
        self.built_environment()

    def built_environment(self):
        self.beta1 = 5
        self.beta2 = -3
        self.sigmaS = 1
        self.sigmaS1 = 0.5
        self.sigmaS2 = 0.3
        self.rmax = 15
        ## Shapley or LOO
        self.pricing_mechanism = 'LOO'

    def step(self, action1, action2):
        X1 = np.random.normal(0, self.sigmaS1 ** .5)
        X2 = np.random.normal(0, self.sigmaS2 ** .5)
        epsilon = np.random.normal(0, self.sigmaS ** .5)
        Y = X1 * self.beta1 + X2 * self.beta2 + epsilon

        X1tilde = X1 + np.random.normal(0, action1 ** .5)
        X2tilde = X2 + np.random.normal(0, action2 ** .5)
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


    #def reset(self):
