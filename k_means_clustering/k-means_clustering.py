# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:13:15 2018

@author: khot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv("C://Users/khot/Desktop/machine learnign/clustering/Mall_Customers.csv")
#creating x
x = ds.iloc[:,[3,4]].values

#choosing no. of clusters using elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init='k-means++',n_init=10 , max_iter=300 ,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11) , wcss)
plt.title("Elbow method")
plt.ylabel("wcss")
plt.xlabel("no. of clusters")
plt.show()

#applying kmeans
kmeans = KMeans(n_clusters=5 , init='k-means++',n_init=10 , max_iter=300 ,random_state=0)
y_means = kmeans.fit_predict(x)

#visualizing 
plt.scatter(x[y_means==0,0] , x[y_means==0,1] , s=100 , c='red' , label='careful')
plt.scatter(x[y_means==1,0] , x[y_means==1,1] , s=100 , c='blue' , label='standard')
plt.scatter(x[y_means==2,0] , x[y_means==2,1] , s=100 , c='green' , label='Target')
plt.scatter(x[y_means==3,0] , x[y_means==3,1] , s=100 , c='cyan' , label='careless')
plt.scatter(x[y_means==4,0] , x[y_means==4,1] , s=100 , c='magenta' , label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s=300 , c='yellow' , label='centroids')
plt.title("clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score (1-100)")
plt.legend()
plt.show()


