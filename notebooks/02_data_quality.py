import pandas as pd
import numpy as np

df = pd.read_excel('../data/raw/Online_Retail.xlsx', parse_dates=["InvoiceDate"])

print("======== Data Quality Report =======")

print("Dataset Shape: ", df.shape)

print("Missing Values: ", df.isnull().sum())

print("Missing values percentage: ", (df.isnull().mean() * 100).round(2))

print("Duplicate Data: ", df.duplicated().sum())

print("Negative Quantities: ", (df["Quantity"] < 0 ).sum())

print("Zero Quantities", (df["Quantity"]== 0).sum())

print("Negative Unit Price: ", (df["UnitPrice"] < 0).sum())

print("Zero Unit Price: ", (df["UnitPrice"] == 0).sum())

df["InvoiceNo"] = df["InvoiceNo"].astype(str)

cancelled = df["InvoiceNo"].str.startswith("C")

print("Cancelled Transactions: ")
print(cancelled.sum())

