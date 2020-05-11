import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

col_names = ['Year','Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Direction']

df = pd.read_csv('Smarket.csv', index_col=None, na_values=['NA'], usecols=col_names)

# Changing to binary input
direction = {'Up': 1,'Down': 0}
df.Direction = [direction[item] for item in df.Direction]

# Only prints the first 5 lines
print(df.head())

# Descriptive statistics
print(df.describe())

# Seperates data into dependent and independent variables
feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
X = df[feature_cols]
y = df.Direction

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X,y)
y_pred=logreg.predict(X)

# Confusion table - the diagonal numbers represents the correct predictions
print(confusion_matrix(y, y_pred))

# Evaluation of model
print("Accuracy:",metrics.accuracy_score(y, y_pred))
print("Precision:",metrics.precision_score(y, y_pred))
print("Recall:",metrics.recall_score(y, y_pred))

# ROC Curve
y_pred_proba = logreg.predict_proba(X)[::,1]
fpr, tpr, _ = metrics.roc_curve(y,  y_pred_proba)
auc = metrics.roc_auc_score(y, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
################################################################
# We try splitting the data set into train and test
df_train = df[df['Year'] < 2005]
df_test = df[df['Year'] >= 2005]


X_train = df_train[feature_cols]
Y_train = df_train['Direction']

X_test = df_test[feature_cols]
Y_test = df_test['Direction']

# Logistic Regression
logit2 = LogisticRegression()
resLogit2 = logit2.fit(X_train, Y_train)

# Predict training set
y_pred2 = resLogit2.predict(X_test)

# Confusion matrix
print(confusion_matrix(Y_test, y_pred2))

# Evaluation of model
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred2))
print("Precision:",metrics.precision_score(Y_test, y_pred2))
print("Recall:",metrics.recall_score(Y_test, y_pred2))

########################################################
# We try to remove useless predictors

feature_cols2 = ['Lag1','Lag2']

X_train2 = df_train[feature_cols2]
Y_train2 = df_train['Direction']

X_test2 = df_test[feature_cols2]
Y_test2 = df_test['Direction']

# Logistic Regression
logit3 = LogisticRegression()
resLogit3 = logit3.fit(X_train2, Y_train2)

# Predict training set
y_pred3 = resLogit3.predict(X_test2)

# Confusion matrix
print(confusion_matrix(Y_test2, y_pred3))

# Evaluation of model
print("Accuracy:",metrics.accuracy_score(Y_test2, y_pred3))
print("Precision:",metrics.precision_score(Y_test2, y_pred3))
print("Recall:",metrics.recall_score(Y_test2, y_pred3))


