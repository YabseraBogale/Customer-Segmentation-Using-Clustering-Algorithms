# Customer-Segmentation-Using-Clustering-Algorithms
#install
pip install pandas pyspark



the code line by line:
# python
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

This part imports the necessary libraries for data manipulation, visualization, and machine learning tasks.
# python
# home_data = pd.read_csv('housing.csv', usecols = ['longitude', 'latitude', 'median_house_value'])
# home_data.head()

This part reads a CSV file named 'housing.csv' into a pandas DataFrame called home_data. It selects only the columns 'longitude', 'latitude', and 'median_house_value' for further analysis. The head() method is used to display the first few rows of the DataFrame.
# python
# sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')

This line creates a scatter plot using seaborn's scatterplot() function. The plot shows the distribution of data points based on their longitude and latitude, with the color (hue) representing the median house value.
# python
# X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

This line splits the data into training and testing sets using train_test_split() from sklearn.model_selection. The features (X) are the latitude and longitude columns, while the target (y) is the median house value column. The test size is set to 0.33, meaning 33% of the data will be used for testing and the remaining 67% for training. The random_state parameter ensures reproducibility of the split.

# python
# X_train_norm = preprocessing.normalize(X_train)
# X_test_norm = preprocessing.normalize(X_test)

This part normalizes the training and testing feature sets using preprocessing.normalize() from sklearn. Normalization is a common data preprocessing step to ensure that all features are on the same scale.
# python
# kmeans = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
# kmeans.fit(X_train_norm)

This part creates a KMeans clustering model with 3 clusters, sets the random_state for reproducibility, and fits the model to the normalized training data using fit().
# python
# sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
# sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'])

These lines create two plots:
A scatter plot of the training data points, colored by their assigned cluster labels from the KMeans model.
A box plot showing the distribution of median house values for each cluster.
# python
# silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

This line calculates the silhouette score for the KMeans clustering using silhouette_score() from sklearn.metrics. The silhouette score measures the quality of the clustering, with values ranging from -1 to 1. Higher values indicate better-defined clusters.
# python
# K = range(2, 8)
# fits = []
# score = []

This part initializes a list of numbers of clusters to try (K), and two empty lists to store the fitted KMeans models (fits) and their corresponding silhouette scores (score).
# python
# for k in K:
    # train the model for current value of k on training data
    model = KMeans(n_clusters = k, random_state = 0, n_init='auto').fit(X_train_norm)

    # append the model to fits
    fits.append(model)

    # Append the silhouette score to scores
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))

This loop iterates over the range of cluster numbers (K). For each value of k:
It creates a KMeans model with k clusters, sets the random_state, and fits the model to the normalized training data.
It appends the fitted model to the fits list.
It calculates the silhouette score for the current model and appends it to the score list.
# python
# sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[0].labels_)
# sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[2].labels_)
# sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = fits[2].labels_)

These lines create three scatter plots of the training data, each colored by the cluster labels assigned by the KMeans models with 2, 4, and 4 clusters, respectively. The purpose is to visually compare the clustering results for different numbers of clusters.
In summary, this code performs KMeans clustering on a dataset containing longitude, latitude, and median house value information. It splits the data into training and testing sets, normalizes the features, and trains KMeans models with different numbers of clusters. The code evaluates the clustering quality using silhouette scores and visualizes the results using scatter plots and box plots.
