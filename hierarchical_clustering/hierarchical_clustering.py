# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:18:14 2018

@author: khot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv("C://Users/khot/Desktop/machine learnign/clustering/Mall_Customers.csv")
#creating x
x = ds.iloc[:,[3,4]].values

#creating dendogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x , method='ward'))
plt.title("dendogram")
plt.xlabel('customers')
plt.ylabel("Euclidian distance")
plt.show()

#fitting clustering
from sklearn.cluster import AgglomerativeClustering
hclst = AgglomerativeClustering(n_clusters=5 , linkage='ward' , affinity='euclidean')
y = hclst.fit_predict(x)

#visualizing 
plt.scatter(x[y==0,0] , x[y==0,1] , s=100 , c='red' , label='careful')
plt.scatter(x[y==1,0] , x[y==1,1] , s=100 , c='blue' , label='standard')
plt.scatter(x[y==2,0] , x[y==2,1] , s=100 , c='green' , label='Target')
plt.scatter(x[y==3,0] , x[y==3,1] , s=100 , c='cyan' , label='careless')
plt.scatter(x[y==4,0] , x[y==4,1] , s=100 , c='magenta' , label='sensible')
plt.title("clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score (1-100)")
plt.legend()
plt.show()
