import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor, summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std



data = pd.read_csv('Wage.csv')

#We change the name of the first column
data = data.rename(columns={'Unnamed: 0': 'Number'})

#We through away the missing values
data = data.dropna()

#Logistic regression
logreg = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data)
estimate = logreg.fit()
print(estimate.summary())

#Creating an array of the age we will do the predictions for
NewAges = pd.DataFrame()
NewAges['age'] = np.arange(data['age'].min(), data['age'].max(), dtype=int)

#Predictions
pred = estimate.get_prediction(NewAges, weights=1)
pred2 = pred.summary_frame(alpha=0.05)
NewAges[['lpred','lstderr','lmean_ci_lower','lmean_ci_upper','lobs_ci_lower','lobs_ci_upper']] = pred2

#Plot the polynomial regression
f, ax = plt.subplots()
ax.scatter(data['age'], data['wage'], facecolor='None', edgecolor='darkgrey', label ="data")
ax.plot(NewAges['age'], NewAges['lpred'], 'g-', label='Prediction')
ax.plot(NewAges['age'], NewAges['lmean_ci_lower'], 'g--', label='Confidence Interval - 95%')
ax.plot(NewAges['age'], NewAges['lmean_ci_upper'], 'g--')


ax.set_xlabel('Age')
ax.set_ylabel('Wage')
plt.title('4th degree Polynomial regression')
plt.show()

#ANOVA

#We fit 5 different models of wage and age
logreg_age1 = smf.ols(formula='wage~age', data=data).fit()
logreg_age2 = smf.ols(formula='wage~age + np.power(age, 2)', data=data).fit()
logreg_age3 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3)', data=data).fit()
logreg_age4 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data).fit()
logreg_age5 = smf.ols(formula='wage~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4) + np.power(age, 5)', data=data).fit()

#We compare these models
anova_age = sm.stats.anova_lm(logreg_age1, logreg_age2, logreg_age3, logreg_age4, logreg_age5)
print(anova_age)

#We now fit 3 different models of wage and education
logreg_edu1 = smf.ols(formula='wage~education + age', data=data).fit()
logreg_edu2 = smf.ols(formula='wage~education + age + np.power(age, 2)', data=data).fit()
logreg_edu3 = smf.ols(formula='wage~education + age + np.power(age, 2) + np.power(age, 3)', data=data).fit()

#We now compare these models
anova_education = sm.stats.anova_lm(logreg_edu1, logreg_edu2, logreg_edu3)
print(anova_education)

#Logistic regression
#Creating binomial values
data['wage_250'] = (data.wage > 250).map({True: 1, False: 0})
logreg_sm = smf.glm(formula='wage_250~age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=data, family=sm.families.Binomial()).fit()
print(logreg_sm.summary())

#Confidence intervals
logreg_pred = logreg_sm.get_prediction(NewAges)
gpred = logreg_pred.summary_frame(alpha=0.05)

NewAges[['gpred','gstderr','gmean_ci_lower','gmean_ci_upper']] = gpred

#Plot 
fg, axg = plt.subplots()
axg.scatter(data['age'], data['wage_250'], marker ='|',color='gray', alpha=0.1)
axg.plot(NewAges['age'], NewAges['gpred'], 'g-')
axg.plot(NewAges['age'], NewAges['gmean_ci_lower'], 'g--', alpha=0.8)
axg.plot(NewAges['age'], NewAges['gmean_ci_upper'], 'g--', alpha=0.8)
axg.set_xlabel('Age')
axg.set_ylabel('Probability')
plt.title('Estimate of fittet probabilities')
plt.show()





