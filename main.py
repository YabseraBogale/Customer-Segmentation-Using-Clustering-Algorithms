import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv("Mall_Customers.csv")


sns.scatterplot(data=df,x='Age',y='Spending_Score',hue = 'Annual_Income_(k$)')

