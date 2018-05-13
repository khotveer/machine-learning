# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:21:58 2018

@author: khot
"""

import pandas as pd
import numpy as np


ds = pd.read_csv("machine-learning/mtcars_regression_analysis/mpg.csv")
#checking the data set dimensions rows , columns 
ds.shape
#getting glance of first three columns
ds.head()
#checking the data type of each feature
ds.dtypes
ds = ds.convert_objects(convert_numeric=True)
#checking for missing values
ds.isnull().sum()
#imputing missing values in ds
ds.fillna(ds.mean(), inplace=True)

x = ds[ds.columns[1:8]]
y = ds['mpg']
y = y.reshape(len(y),1)

#using scatter plot to see linear relationship
import matplotlib.pyplot as plt
for i in range(0,7):
    plt.scatter(x[x.columns[i]] , y)
    plt.xlabel(x.columns[i])
    plt.ylabel('mpg')
    plt.show()

#splitting data into train and test
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=0)
#fitting the line (all in regression method)
from sklearn import linear_model
lr = linear_model.LinearRegression(normalize=True)
lr.fit(x_train , y_train)
#predicting the vlues
y_predicted = lr.predict(x_test)
#r-squared
from sklearn.metrics import r2_score 
print("r-squared ",r2_score(y_test , y_predicted))

#stepwise regression for getting more optimal model
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((398,1)).astype(int)  , values=x, axis=1)

x_opt = x[:,[0,1,2,3,4,5,6,7]]
model1 = sm.OLS(endog=y , exog=x_opt).fit()

print(model1.summary())
print("there is no p-value > 0.05 so we choose this model as final model with adj R-squared ", model1.rsquared_adj)
