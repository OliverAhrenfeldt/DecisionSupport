import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils.random as sk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#read file and make np array
df = pd.read_csv('5.3.1/Auto.csv')
dfnp = np.array(df)

#iterate through the data and delete all rows where horsepower is '?'
#we cannot use rows where horsepower is not specified

deletearray = []
for index, hp in enumerate(df['horsepower'], start=1):
    if hp == '?':
        deletearray.append(index-1)

dfnp = np.delete(dfnp, deletearray, axis=0)

# picks 196 random indexes from 392 numbers
train = sk.sample_without_replacement(392, 196)

#we choose mpg as x and horsepower as y
x = dfnp[:, 0].reshape((-1, 1))
y = dfnp[:, 3].reshape((-1, 1))

#we split data into 2 for train- and test purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

#we make a linear regression on the train set
model = LinearRegression()
model.fit(x_train, y_train)

print(model.coef_)
print(model.intercept_)

#we use our model, predicting y calues from the test-x values
y_pred = model.predict(x_test)

#making arrays from matrixes
ytest=np.squeeze(np.asarray(y_test).reshape((1, -1)))
ypred=np.squeeze(np.asarray(y_pred).reshape((1, -1)))

#converting stringlist to intlist
ytest = list(map(int, ytest))

#making a dataframe comparison between the actual and the predicted y values
pr = pd.DataFrame({'Actual': ytest, 'Predicted': ypred})
pr1 = pr.head(25)
pr1.plot.bar()

plt.savefig('5.3.1/valid.png')
plt.show()


