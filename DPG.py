import numpy as np
import itertools
import math


class DataProvider:
    def __init__(self, mu,  sigma_square, cost_function, agent):
        self.mu = mu
        self.sigma_square = sigma_square
        self.cost_function = cost_function
        self.agent = agent

        self.s_square = None

        self.actions = []
        self.rewards = []

    def choose_action(self, episode):
        # TODO: integrate agent
        self.s_square = self.agent.choose_action(episode)
        # self.s_square = np.random.uniform(0, self.sigma_square ** .5)
        self.actions.append(self.s_square)
        return self.s_square

    def receive_payment(self, payment, i):
        # costs = eval(self.cost_function, {"s_square": self.s_square, "sigma_square": self.sigma_square})
        # costs = Training.cost_function(sigma_square=self.sigma_square, s_square=self.s_square)
        costs = self.cost_function.evaluate({"s_square": self.s_square, "sigma_square": self.sigma_square})
        reward = payment - costs
        self.rewards.append(reward)
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
                 data_consumer: DataConsumer, data_set, sigma_square=1):
        self.data_set = data_set
        self.data_provider_list = data_provider_list
        self.analytics_service_provider = analytics_service_provider
        self.data_consumer = data_consumer

        self.sigma_square = sigma_square

    def step(self, i):
        n = len(self.data_provider_list)
        # Data Providers decide actions
        actions = np.zeros(n)
        for j in range(n):
            actions[j] = self.data_provider_list[j].choose_action(i)

        # The predictors take on their values
        # TODO: read values from data set
        predictor = np.zeros(n)
        if self.data_set:
            predictor = self.data_set.get_predictor(i)
        else:
            for j in range(n):
                predictor[j] = np.random.normal(loc=self.data_provider_list[j].mu,
                                                scale=self.data_provider_list[j].sigma_square ** .5)

        # Perform measurements
        measurements = np.zeros(n)
        for j in range(n):
            measurements[j] = self.data_provider_list[j].perform_measurement(predictor[j])

        # Data interpretation by analytics service provider
        estimated_information = self.analytics_service_provider.interpret(measurements)

        # Target variable takes on its value
        # TODO: read values from data set
        if self.data_set:
            information = self.data_set.get_information(i)
        else:
            epsilon = np.random.normal(0, self.sigma_square ** .5)
            information = sum(measurements * self.analytics_service_provider.beta) + epsilon

        # data consumer receives revenue

        # payments are calculated and paid out to data consumer
        # Due to normal distribution a Data Provider will never be completely inactive
        for j in range(n):
            payment = self.data_consumer.calculate_payments(information, estimated_information, j, n)
            self.data_provider_list[j].receive_payment(payment, i)
