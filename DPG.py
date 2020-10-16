import numpy as np


class DataProvider:
    def __init__(self, mu,  sigma_square, cost_function, algorithm="hoo"):
        self.mu = mu
        self.sigma_square = sigma_square
        self.algorithm = algorithm
        self.cost_function = cost_function

        self.s_square = mu

    def choose_action(self):
        # TODO: integrate agent
        self.s_square = np.random.uniform(0, self.sigma_square ** .5)
        return self.s_square

    def perform_measurement(self, predictor):
        measurement = predictor + np.random.normal(self.mu, self.s_square ** .5)
        return measurement


class DataConsumer:
    def __init__(self, y, pricing_mechanism='loo'):
        self.pricing_mechanism = pricing_mechanism


class AnalyticsServiceProvider:
    def __init__(self, beta):
        self.beta = np.asanyarray(beta)

    def interpret(self, measurements):
        return measurements * self.beta


class DataProvisionGame:
    def __init__(self, data_provider_list: list[DataProvider], analytics_service_provider: AnalyticsServiceProvider,
                 data_consumer: DataConsumer):
        self.data_provider_list = data_provider_list
        self.analytics_service_provider = analytics_service_provider
        self.data_consumer = data_consumer

    def step(self):
        n = len(self.data_provider_list)
        # Data Providers decide actions
        actions = np.zeros(n)
        for j in range(n):
            actions[j] = self.data_provider_list[j].choose_action()

        # The predictors take on their values
        # TODO: read values from data set
        predictor = np.zeros(n)
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
    p3 = Agent(beta=2, sigma_square=0.8, mu=0)
    agents = [p0, p1]

    # hyperparameter
    rho = 0.998
    v1 = 1.5

    dpg = DPG(agents, pricing_mechanism='LOO')
    print("training ...")
    dpg.play(v1=v1, rho=rho, n=rounds)