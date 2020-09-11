import json
import os
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
from sklearn.cluster import KMeans
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import silhouette_score


def Silhouette_index(df, no_of_clusters,cluster_labels):

    X = df[['numbers']]
    for n_clusters in no_of_clusters:
        silhouette_avg = silhouette_score(X, cluster_labels)
        df['clusterNumber'] = cluster_labels
        #print(df)
        print("Silhouette Score", " :", silhouette_avg)


#Data Initialization
data1 = [2, 4, 6, 7, 8, 9, 10, 11, 13]
data2 = [1, 3, 5, 6, 7, 8]

#Cluster fixation
cluster_labels1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
cluster_labels2 = [0, 0, 0, 1, 1, 1]

#Number of Cluster
no_of_clusters1 = [3]
no_of_clusters2 = [2]

#Create Dataframe
df = pd.DataFrame(data1, columns=['numbers'])
df2 = pd.DataFrame(data2, columns=['numbers'])

#Silhouette Index Calculation
print("First Data")
Silhouette_index(df, no_of_clusters1,cluster_labels1)
print("========================================")
print("Second Data")
Silhouette_index(df2, no_of_clusters2,cluster_labels2)




