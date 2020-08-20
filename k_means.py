import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import statistics as st
import stat as stt
import csv
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;

#Load Data
dt = pd.read_csv("./Cluster/conservative.csv")
df2 = pd.read_csv("./Cluster/181_hedonists.csv")
df = pd.read_csv("./Hedonism_cook_dendo2.csv")
#centers.reshape(1, -1)

#Predetermined Cluster Centers from Two Step Cluster
centers = np.array([[.55802263,.82198811,.91454810,.55311750,.53383340],
                    [.22113488,.52658067,.68526141,.29774772,.36226243],
                    [.50526616,.55504691,.81730927,.29774772,.31918362],
                    [.45377774,.38088997,66315407,.16667629,.16326850],
                    [.29361346,.27282872,.45742593,.07993467,.11764417]],np.float64)

					
					
hedonism_df = df[['Conservation','Openness to change', 'Hedonism','Self-transcendence', 'Self-enhancement']]

dt['Hedonism'] = df['Hedonism']

#CSV Conversion
def tocsv(name):
    to_csv = df.to_csv('./Cluster/'+name+'.csv', index=None, header=True)

#Simple Kmeans cluster on whole dataset on 5 different attribute	
def cluster_big5():
    kmeans = KMeans(n_clusters=3)
    df['Conservation_Cluster'] = kmeans.fit_predict(df[['Conservation']])
    df['Openness-to-change_Cluster'] = kmeans.fit_predict(df[['Openness to change']])
    df['Self-enhancement_Cluster'] = kmeans.fit_predict(df[['Self-enhancement']])
    df['Self-transcendence_Cluster'] = kmeans.fit_predict(df[['Self-transcendence']])
    df['Hedonism_Cluster'] = kmeans.fit_predict(df[['Hedonism']])

#Parallel KMeans Cluster
def clusterall():
    #Kmeans on 2 parallels where cluster number is defined and centers are predetermined
    kmeans = KMeans(n_clusters=5,init=centers,n_init=1, n_jobs=2)
    #Save cluster 
    df['cluster'] = kmeans.fit_predict(hedonism_df)
    
	#Plotting of computed clusters based on Hedonism 
	
    ls = ['Self-Transcendence','Self-Enhancement', 'Conservation', 'Openness to change', 'Hedonism']
    y_hc = df['cluster']
    X = np.array(hedonism_df)
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=20, c='cyan')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=20, c='green')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=20, c='red')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=20, c='blue')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=20, c='magenta')
    plt.legend(loc='upper center',  bbox_to_anchor=(1.17, 0.7), fontsize = 8, ncol=1)

    df.to_csv("Datapoints_with_cluster.csv")
    plt.show()

    cluster_labels = df['cluster'].tolist()
    silhouette_avg = silhouette_score(x, cluster_labels)

    print(silhouette_avg)


#Calculating Silhouette Index
def Silhouette_index():
    #Load Data
    df_tech = pd.read_csv("./Cluster/tech_values.csv")
    df_movie = pd.read_csv("./Cluster/movie_values.csv")
    df_newHedonists = pd.read_csv("169_hedonists.csv")

    no_of_clusters = [2,3,4,5]
    print("Hedonists")
    X = df_newHedonists[['Hedonism']]
    for n_clusters in no_of_clusters:
        cluster = KMeans(n_clusters=n_clusters)
        cluster_labels = cluster.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)

        print(n_clusters, " :", silhouette_avg)


    print("Tech")
    X = df_tech[['Hedonism']]
    for n_clusters in no_of_clusters:
        cluster = KMeans(n_clusters=n_clusters)
        cluster_labels = cluster.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)

        print(n_clusters," :", silhouette_avg)

    print("Movie")
    X = df_movie[['Hedonism']]
    for n_clusters in no_of_clusters:
        cluster = KMeans(n_clusters=n_clusters)
        cluster_labels = cluster.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)

        print(n_clusters, " :", silhouette_avg)


Silhouette_index()
clusterall()

