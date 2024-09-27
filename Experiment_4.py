import pandas as pd
data = pd.read_csv("data.csv")

print("Number of rows:", data.shape[0]) 
print("Number of columns:", data.shape[1]) 
print("\nFirst five rows of the dataset:") 
print(data.head())
print("\nSize of the dataset:", data.size)

print("Number of missing values in each column:") 
print(data.isnull().sum())

print("\nSum of numerical columns:") 
print(data.sum(numeric_only=True))

print("Average of numerical columns:") 
print(data.mean(numeric_only=True))

print("Minimum values of numerical columns:") 
print (data.min (numeric_only=True))

print("\nMaximum values of numerical columns:") 
print(data.max(numeric_only=True))

data.to_csv('data.csv', index=False)
