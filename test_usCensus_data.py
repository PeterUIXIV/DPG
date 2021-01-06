import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


DATA_PATH = r'D:\workspaces\datasets\USCensus1990\USCensus1990.data.txt'
df = pd.read_csv(DATA_PATH)
print("df read")
mean = df.mean()
std_dev = df.std()
var = df.var()
print(f"mean: {mean}, std_dev = {std_dev}, var: {var}")

corr = df.corr()
print("corr matrix calculated")
'''
f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number)
plt.xticks(range(corr.shape[1]), corr.columns, fontsize=12, rotation=45)
plt.yticks(range(corr.shape[1]), corr.columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

plt.show()
'''

au_corr = corr.abs().unstack()

print("after au_corr")
pairs_to_drop = set()
cols = df.columns

print("before for")
for i in range(0, df.shape[1]):
    for j in range(0, i+1):
        pairs_to_drop.add((cols[i], cols[j]))

print("after for")
au_corr = au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False)

print(au_corr[0:40])


'''
with pd.option_context('display.max_rows', None, 'display.max_columns',  None):
...     print(corr.loc['iMilitary'].abs().sort_values(ascending=False))
'''
# iMobillim, iWork89, dAge
X = df[['iMobillim', 'iWork89']]
Y = df['iMilitary']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print(regr.score(X, Y))
print(regr.intercept_)
print(regr.coef_)
plt.scatter(df['iMobillim'], df['iMilitary'], color='red')
