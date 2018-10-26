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

df1 = pd.read_csv("adrb2_properties.csv")   #reading the csv file

df2 = df1.iloc[:,2:55]  #extracting the features from the dataframe
print(df2.shape)    #print statement for debugging

#Data pre-processing	NORMALIZATION

x = df2.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()   #scaling the feature range
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled) #scaled feature dataframe


reduced_data = PCA(n_components=23).fit_transform(df)   #choosing number of principal components as 23
silhouette = []
daviesBouldin = []
distortions = []
K = range(2,30)
for k in K:
	kmeans = KMeans(n_clusters=k).fit(reduced_data)
	kmeans.fit(reduced_data)
	distortions.append(sum(np.min(cdist(reduced_data,kmeans.cluster_centers_, 'euclidean'), axis=1))/ reduced_data.shape[0])
	labels = kmeans.labels_
	sh = metrics.silhouette_score(reduced_data, labels, metric='euclidean')
	db = davies_bouldin_score(reduced_data, labels)
	silhouette.append(sh)
	daviesBouldin.append(db)

# Plot the elbow graph
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.figure()

#Calculate the silhouette index for each k
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k-number')
plt.ylabel('Silhouette score')
plt.title('Silhouette scores for varying k')
plt.figure()

#Calculate the DB index for each k
plt.plot(K, daviesBouldin, 'bx-')
plt.xlabel('k-number')
plt.ylabel('Davies-Bouldin score')
plt.title('Davies-Bouldin scores for varying k')

plt.show()
