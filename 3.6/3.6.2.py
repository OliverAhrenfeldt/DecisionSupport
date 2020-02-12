import csv
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# Reading the CSV file
with open('3.6/Boston.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    x = []
    y = []

    # Placing the relevant columns in two arrays
    for row in reader:
        if row[0] != '':
            x.append(float(row[13]))
            y.append(float(row[14]))

# Plot
mpl.scatter(x, y, marker='+')
mpl.xlabel('lstat')
mpl.ylabel('medv')

# Linear regression
# https://seaborn.pydata.org/generated/seaborn.regplot.html
sb.regplot(x, y, color='red', scatter=False)
mpl.show()


# # Indlæser dataset
# data = pd.read_csv('3.6/Boston.csv')

# # # Konverter data til et numpy array
# x = data.iloc[:, 13].values.reshape(-1, 1)
# y = data.iloc[:, 14].values.reshape(-1, 1)

# # Beregning af fittede værdier
# # https://towardsdatascience.com/linear-regression-in-6-lines-of-python-5e1d0cd05b8d
# linear_regressor = LinearRegression()
# linear_regressor.fit(x, y)
# Y_pred = linear_regressor.predict(x)

# # Plot
# data.plot(x='lstat', y='medv', style='o')
# mpl.xlabel('Lower status of the population [pct]')
# mpl.ylabel('Median value of owner-occupied homes [1000 $]')
# mpl.plot(x, Y_pred, color='red')
# mpl.show()

# # Forskellige informationer om data og fittet
# f_reg = f_regression(x, y, center=True)
# F = float(f_reg[0])
# p = float(f_reg[1])

# print('F-value: ', F)
# print('P-value of F-score: ', p)
