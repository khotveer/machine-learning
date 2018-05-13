# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:05:21 2018

@author: khot
"""

import pandas as pd
import numpy as np

#importing dataset from sklearn library
from sklearn import datasets
data = datasets.load_boston()
ds = pd.DataFrame(data.data , columns=data.feature_names)
#checking the data set dimensions rows , columns 
ds.shape
#getting glance of first three columns
ds.head()
#checking the data type of each feature
ds.dtypes
#checking null values in ds
ds.isnull().sum().sum()

x = ds.iloc[:,:]
x.shape
y = data.target
y =y.reshape(len(y) , 1)


#scatter plots to see corelation between all Xn and y
import matplotlib.pyplot as plt
for i in range(1 ,14):
    plt.scatter(x[:,i] , y)
    plt.xlabel("x"+str(i))
    plt.ylabel("y")
    plt.show()
    
#splitting datasets into train and test
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state=0)

#fitting the line (all in regression method)
from sklearn import linear_model
lr = linear_model.LinearRegression(normalize=True)
lr.fit(x_train , y_train)

#predict y using x_test
predicted_y = lr.predict(x_test)

#checking r-squared
import sklearn.metrics as m
print("r-squared : ",m.r2_score(y_true=y_test , y_pred=predicted_y))

#stepwise regression using backward elimination for optimal model
#where level of significane =0.05
#eliminating feature if p>level of significance
import statsmodels.formula.api as sm
x = np.append(arr=np.ones((506,1)).astype(int)  , values=x, axis=1)

x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
model1 = sm.OLS(endog=y , exog=x_opt).fit()
print(model1.summary())

x_opt = x[:,[0,1,2,3,4,5,6,8,9,10,11,12,13]]
model2 = sm.OLS(endog=y , exog=x_opt).fit()
print(model2.summary())

x_opt = x[:,[0,1,2,4,5,6,8,9,10,11,12,13]]
model3 = sm.OLS(endog=y , exog=x_opt).fit()
print(model3.summary())

print("models and their adjusted R-squared : ")
print("model1 : ",model1.rsquared_adj)
print("model2 : ",model2.rsquared_adj)
print("model3 : ",model3.rsquared_adj)

print("so from above models most accurate model is model3 with ajd r-squared value = ",model3.rsquared_adj)