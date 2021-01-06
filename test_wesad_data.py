import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


DATA_PATH = r'D:\workspace\datasets\WESAD\S2\S2_E4_Data\ACC.csv'
df = pd.read_csv(DATA_PATH, skiprows=1)
df.columns = ['1', '2', '3']

mean = df.mean()
std_dev = df.std()
var = df.var()