import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn import metrics
from sklearn.metrics import pairwise_distances

from sklearn.metrics import davies_bouldin_score

from sklearn.decomposition import PCA   #importing PCA library
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import FastICA

df1 = pd.read_csv("adrb2_properties.csv")   #reading the csv file

df2 = df1.iloc[:,2:55]  #extracting the features from the dataframe
print(df2.shape)    #print statement for debugging

#Data pre-processing	NORMALIZATION

x = df2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()   #scaling the feature range
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled) #scaled feature dataframe


#This code snippet is to implement PCA
#***************************************************************************************************
sum_comps = []
comps = []
for i in range(1,54):
	x = 0
	pca = PCA(n_components=i)
	X_PCA = pca.fit(df).transform(df)
	x = np.sum(pca.explained_variance_ratio_)
	print(i, x, x*100)
	sum_comps.append(x*100)
	comps.append(i*100/58)

print("PCA variance of components:",pca.explained_variance_ratio_)
print(df.shape)
plt.bar(comps,sum_comps)
plt.xlabel('Precentage of Features')
plt.ylabel('Percentage of information retained')
plt.title('Feature Retention vs Variance Ratio')
plt.show()


pca = PCA(n_components=23)
X_PCA = pca.fit(df).transform(df)
