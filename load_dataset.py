import pandas as pd
from sklearn import linear_model


class DataSet:
    def __init__(self, path, predictor, target):
        self.df = pd.read_csv(path)
        # self.corr = df.corr()
        self.X = self.df[predictor]
        self.Y = self.df[target]
        regr = linear_model.LinearRegression()
        regr.fit(self.X, self.Y)
        self.coef = regr.coef_
        self.var = self.df.var()[predictor]
        print(f"coef: {self.coef}")
        print(f"var: {self.var}")
        print("data set loaded")

    def get_predictor(self, i):
        return self.X.loc[i]

    def get_information(self, i):
        return self.Y.loc[i]


