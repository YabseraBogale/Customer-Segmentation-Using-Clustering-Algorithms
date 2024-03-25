import pandas as pd


df=pd.read_csv("Mall_Customers.csv")


print(df.describe())
print(df.info())