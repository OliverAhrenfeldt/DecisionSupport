import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

# Inspiration: https://datatofish.com/multiple-linear-regression-python/

data = pd.read_csv('3.6/Boston.csv')
df = pd.DataFrame(data, columns=['age', 'medv', 'lstat'])

# SIMPLE LINEAR REGRESSION
X_sim = df['lstat'].values.reshape(-1, 1)
Y_sim = df['medv']

regr = linear_model.LinearRegression()
regr.fit(X_sim, Y_sim)

print('Intercept with simple linear regression: ', regr.intercept_)
print('Coefficients with simple linear regression: ', regr.coef_)

# MULTIPLE LINEAR REGRESSION
# Here we have two variables, as we are now considering investigating whether
# age is a contributing factor
X_mul = df[['age', 'lstat']]
Y_mul = df['medv']

regr = linear_model.LinearRegression()
regr.fit(X_mul, Y_mul)

print('Intercept with multiple linear regression: ', regr.intercept_)
print('Coefficients with multiple linear regression: ', regr.coef_)
