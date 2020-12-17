import numpy as np
import itertools
import math

debug = False


class DataProvider:
    def __init__(self, mu,  sigma_square, cost_function, agent, setup):
        self.mu = mu
        self.sigma_square = sigma_square
        self.cost_function = cost_function
        self.agent = agent
        self.setup = setup

        self.s_square = None

        self.actions = []
        self.rewards = []

    def choose_action(self, episode):
        self.s_square = self.agent.choose_action(episode)

        # rescale s_square to from interval [0, 1] to [lower, sigma_square]
        if self.setup == 'eps_greedy':
            pass
        else:
            lower = 0.0
            self.s_square = (self.sigma_square - lower) / (1 - 0) * (self.s_square - 1) + self.sigma_square

        self.actions.append(self.s_square)
        return self.s_square

    def receive_payment(self, payment, i, k):
        # costs = eval(self.cost_function, {"s_square": self.s_square, "sigma_square": self.sigma_square})
        # costs = Training.cost_function(sigma_square=self.sigma_square, s_square=self.s_square)
        costs = self.cost_function.evaluate({"s_square": self.s_square, "sigma_square": self.sigma_square})
        reward = payment - costs
        self.rewards.append(reward)

        if debug:
            print(f"Payment: {payment}, costs: {costs}, reward: {reward}, s_square: {self.s_square}")

        # logistic function with logistic growth k and midpoint x_0
        if self.setup == 'stretch':
            reward = reward / 5
        elif self.setup == 'logistic':
            x_0 = 0
            if (reward - x_0) < 0:
                reward = 1 - 1 / (1 + math.exp(k * (reward-x_0)))
            else:
                reward = 1 / (1 + math.exp(-k * (reward-x_0)))
        elif self.setup == 'bounded':
            high = 30
            low = -20
            if reward >= high:
                reward = 1
            elif reward <= low:
                reward = 0
            else:
                reward = (1 - 0) / (high - low) * (reward - high) + 1
        elif self.setup == 'eps_greedy' or self.setup == 'real':
            pass

        self.agent.learn(reward, i)

    def perform_measurement(self, predictor):
        measurement = predictor + np.random.normal(self.mu, self.s_square ** .5)
        return measurement


class DataConsumer:
    def __init__(self, pricing_mechanism='loo', r_max=15):
        self.r_max = r_max
        self.pricing_mechanism = pricing_mechanism

    def calculate_payments(self, information, estimated_information, j, n):
        if self.pricing_mechanism == 'loo':
            x = list(range(n))
            x.remove(j)
            p = self.v(range(n), estimated_information, information) - self.v(x, estimated_information, information)
            return p
        elif self.pricing_mechanism == 'shapley':
            p = 0
            for perm in itertools.permutations(range(n)):
                for i in perm:
                    if perm[i] == j:
                        p += self.v(perm[:i + 1], estimated_information, information) - \
                             self.v(perm[:i], estimated_information, information)
            return p / math.factorial(n)
        else:
            raise Exception("No valid pricing mechanism")

    def v(self, indices, estimated_information, information):
        sum_y_tilde_s = 0
        for i in indices:
            sum_y_tilde_s += estimated_information[i]
        return self.r_max - (sum_y_tilde_s - information) ** 2


class AnalyticsServiceProvider:
    def __init__(self, beta):
        self.beta = np.asarray(beta)

    def interpret(self, measurements):
        return measurements * self.beta


class DataProvisionGame:
    def __init__(self, data_provider_list: list, analytics_service_provider: AnalyticsServiceProvider,
                 data_consumer: DataConsumer, data_set, sigma_square=1, logistic_growth=0.2):
        self.data_set = data_set
        self.data_provider_list = data_provider_list
        self.analytics_service_provider = analytics_service_provider
        self.data_consumer = data_consumer
        self.logistic_growth = logistic_growth

        self.sigma_square = sigma_square

    def step(self, i):
        n = len(self.data_provider_list)
        # Data Providers decide actions
        actions = np.zeros(n)
        for j in range(n):
            actions[j] = self.data_provider_list[j].choose_action(i)

        # The predictors take on their values
        predictor = np.zeros(n)
        if self.data_set:
            predictor = self.data_set.get_predictor(i)
            if debug:
                print(f"Predictors: {predictor}")
        else:
            for j in range(n):
                predictor[j] = np.random.normal(loc=self.data_provider_list[j].mu,
                                                scale=self.data_provider_list[j].sigma_square ** .5)
            if debug:
                predictor = np.array([-0.473797659, 0.121024593])

        # Perform measurements
        measurements = np.zeros(n)
        for j in range(n):
            measurements[j] = self.data_provider_list[j].perform_measurement(predictor[j])

        if debug:
            if self.data_set:
                print(f"Measurements: {measurements}")
            else:
                measurements = np.array([-0.426597148, 0.650116791])

        # Data interpretation by analytics service provider
        estimated_information = self.analytics_service_provider.interpret(measurements)
        if debug:
            print(f"estimates Ytilde_1: {estimated_information[0]} Ytilde_2: {estimated_information[1]}")

        # Target variable takes on its value
        if self.data_set:
            information = self.data_set.get_information(i)
        else:
            epsilon = np.random.normal(0, self.sigma_square ** .5)
            if debug:
                epsilon = 0.10286918
            information = sum(predictor * self.analytics_service_provider.beta) + epsilon
        if debug:
            print(f"Information Y: {information}")

        # data consumer receives revenue

        # payments are calculated and paid out to data consumer
        # Due to normal distribution a Data Provider will never be completely inactive
        for j in range(n):
            payment = self.data_consumer.calculate_payments(information, estimated_information, j, n)
            if debug:
                print(f"payment: {payment}")
            self.data_provider_list[j].receive_payment(payment, i, self.logistic_growth)
