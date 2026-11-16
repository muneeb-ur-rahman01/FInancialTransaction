import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("../data/raw/online_Retail.xlsx")

print(df.head())
print(df.info())
print(df.columns.to_list())
print("Dataset Shape: " , df.shape)
print(df.describe(include="all"))
print(df.duplicated().sum())
