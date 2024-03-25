import pyspark as spark
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# Load the customer data into a DataFrame
customer_data = spark.read.csv("Mall_Customers.csv", header=True, inferSchema=True)

# Select the relevant features for clustering
feature_cols = ['CustomerID', 'Genre', 'Age']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(customer_data)

# Train the K-means clustering model
kmeans = KMeans(k=5, seed=42)
model = kmeans.fit(data)

# Make predictions
predictions = model.transform(data)

# Evaluate the clustering results using WCSS
evaluator = ClusteringEvaluator()
wcss = evaluator.evaluate(predictions)
print("Within-Cluster Sum of Squares (WCSS):", wcss)
