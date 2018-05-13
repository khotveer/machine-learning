# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:09:50 2018

@author: khot
"""
import numpy as np
import pandas as pd

#importing the data set
ds  = pd.read_csv("50_Startups.csv")
#checking the data set dimensions rows , columns 
ds.shape
#getting glance of first three columns
ds.head()
#checking the data type of each feature
ds.dtypes

#here state is catagorical feature
#checking no. of unique catagories
ds['State'].value_counts()
#checking null values in ds
ds.isnull().sum().sum()

x = ds.iloc[:,:-1].values
y = ds.iloc[:,4].values

#handling categorical variables
#creating dummy variables
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
encode_x = LabelEncoder()
x[:,3] = encode_x.fit_transform(x[:,3])
onhot = OneHotEncoder(categorical_features=[3])
x = onhot.fit_transform(x).toarray()

#checking the relationship between all Xn with y using scatter plots
import seaborn as sns
ds.columns
sns.lmplot(x ="R&D Spend" , y="Profit", hue="State"  , data=ds , fit_reg = False)
sns.lmplot(x ="Administration" , y="Profit", hue="State"  , data=ds , fit_reg = False)
sns.lmplot(x ="Marketing Spend" , y="Profit", hue="State"  , data=ds , fit_reg = False)

#avoiding dummy variables trap
x = x[:,1:]
#splitting dataset into train and test
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.20 , random_state=0)

#fitting the line
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(x_train , y_train)

#predict y using x_test
predicted_y = lr.predict(x_test)
#checking r-squared
import sklearn.metrics as m
print("r-squared : ",m.r2_score(y_true=y_test , y_pred=predicted_y))

#stepwise regresson using backwork elimination Method
#we choose level of significance  = 0.07
#we eliminate feature if p-value> level of significance

import statsmodels.formula.api as sm
x = np.append(arr=np.ones((50,1)).astype(int)  , values=x, axis=1)

x_opt = x[:,[0,1,2,3,4,5]]
model1 = sm.OLS(endog=y , exog=x_opt).fit()
print(model1.summary())

x_opt = x[:,[0,1,3,4,5]]
model2 = sm.OLS(endog=y , exog=x_opt).fit()
print(model2.summary())

x_opt = x[:,[0,3,4,5]]
model3 = sm.OLS(endog=y , exog=x_opt).fit()
print(model3.summary())

x_opt = x[:,[0,3,5]]
model4 = sm.OLS(endog=y , exog=x_opt).fit()
print(model4.summary())

print("models and their adjusted R-squared : ")
print("model1 : ",model1.rsquared_adj)
print("model2 : ",model2.rsquared_adj)
print("model3 : ",model3.rsquared_adj)
print("model4 : ",model4.rsquared_adj)

print("so from above models most accurate model is model3 with ajd r-squared value = ",model3.rsquared_adj)