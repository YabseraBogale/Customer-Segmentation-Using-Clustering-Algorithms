import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Mall_Customer.csv')
df.head(10)
df.shape
df.info()
x = df.iloc[:, [3,4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11);
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('wcss values')
plt.show()

kmeansmodel = kmeans(n_clusters = 5, init='k-means++', random_state=0)

y_kmeans = kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s= 80, c= "red", label='customer 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s= 80, c= "blue", label='customer 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s= 80, c= "yellow", label='customer 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s= 80, c= "cyan", label='customer 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s= 80, c= "black", label='customer 5')
plt,scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c= 'magenta' label= 'centroid')
plt.title('clusters of customers')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()