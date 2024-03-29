import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("Mall_Customers.csv")



print(df.isnull().sum())