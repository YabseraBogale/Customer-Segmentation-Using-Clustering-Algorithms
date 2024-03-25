import pandas as pd


df=pd.read_csv("purchase data.csv")


print(df.describe())
print(df.info())