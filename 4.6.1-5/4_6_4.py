import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Read data
col_names = ['Count','Year','Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Direction']
df = pd.read_csv('Smarket.csv', index_col=None, na_values=['NA'], usecols=col_names)

# Changing to binary input
direction = {'Up': 1,'Down': 0}
df.Direction = [direction[item] for item in df.Direction]

# Selecting only Lag1 and Lag2
feature_cols = ['Lag1','Lag2']

# Splitting into testing and training set
df_train = df[df['Year'] < 2005]
df_test = df[df['Year'] >= 2005]
X_train = df_train[feature_cols]
Y_train = df_train['Direction']

X_test = df_test[feature_cols]
Y_test = df_test['Direction']

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
resqda = qda.fit(X_train,Y_train)
y_pred = resqda.predict(X_test)

# Descriptive statistics 
# Prior probabilities
print("\nPrior probabilities")
print(resqda.classes_)
print(resqda.priors_)
# Mean
print("\nGroup mean")
print(resqda.means_)

#Confusion matrix
print("\nConfusion matrix")
print(confusion_matrix(Y_test, y_pred))

# Evaluation of model
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Precision:",metrics.precision_score(Y_test, y_pred))
print("Recall:",metrics.recall_score(Y_test, y_pred))