import pandas as pd
import numpy as np

Portfolio = pd.read_csv('5.3.4/Portfolio.csv')
Portfolio.describe()
df = pd.DataFrame(Portfolio, columns=['X', 'Y'])


def alpha(data, indices):
    colx = data['X'].tolist()
    coly = data['Y'].tolist()

    X = [colx[i] for i in indices]
    Y = [coly[i] for i in indices]

    return ((np.var(Y)-np.cov(X, Y))/(np.var(X)+np.var(Y)-2*np.cov(X, Y)))


# Without bootstrap
print('Without Bootstrap: \n', alpha(df, range(1, 100)))

# With bootstrap
print('With Bootstrap: \n', alpha(
    df, np.random.choice(100, size=100, replace=True, p=None)))
