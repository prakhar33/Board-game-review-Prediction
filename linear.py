#Data preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
dataset=pd.read_csv('games.csv')
dataset=dataset[dataset['users_rated']>0]
dataset=dataset.dropna(axis=0)
"""plt.hist(dataset['average_rating'])
plt.show()"""
corrmat=dataset.corr()
fig=plt.figure()
sns.heatmap(corrmat,vmax=2,square=True)
plt.show()
X=dataset.iloc[:,[3,4,5,6,7,8,9,10,13,14,15,16,17,18,19]].values
y=dataset.iloc[:,11].values

#Splitting in training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Simple linear regression model to training set

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
error=mean_squared_error(y_pred,y_test)

