import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('clean_data.csv')

# Step 2: Simulate missing values by randomly assigning some cells to NaN
data_missing = data.copy()
data_missing.loc[0, 'Age'] = np.nan  # Simulating a missing value in 'Age'
data_missing.loc[2, 'Salary'] = np.nan  # Simulating a missing value in 'Salary'

# Step 3: Handling Missing Values
# Filling missing numerical values with the median (avoiding chained assignment warning)
data_missing['Age'] = data_missing['Age'].fillna(data_missing['Age'].median())
data_missing['Salary'] = data_missing['Salary'].fillna(data_missing['Salary'].median())

# Step 4: Detecting Outliers using the IQR (Interquartile Range) method
Q1 = data_missing[['Age', 'Salary']].quantile(0.25)  # 25th percentile
Q3 = data_missing[['Age', 'Salary']].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range

# Identifying outliers using the 1.5 * IQR rule
outliers = ((data_missing[['Age', 'Salary']] < (Q1 - 1.5 * IQR)) | 
            (data_missing[['Age', 'Salary']] > (Q3 + 1.5 * IQR)))

# Step 5: Capping outliers: Replace outliers with the max/min allowed within the 1.5 * IQR range
data_no_outliers = data_missing.copy()

for column in ['Age', 'Salary']:
    lower_bound = Q1[column] - 1.5 * IQR[column]
    upper_bound = Q3[column] + 1.5 * IQR[column]
    
    # Replace values below the lower bound with the lower bound
    data_no_outliers[column] = np.where(data_no_outliers[column] < lower_bound, lower_bound, data_no_outliers[column])
    
    # Replace values above the upper bound with the upper bound
    data_no_outliers[column] = np.where(data_no_outliers[column] > upper_bound, upper_bound, data_no_outliers[column])

# Step 5: Comparing differences
# Finding changes after missing value handling
missing_value_changes = data_missing.compare(data)

# Finding changes after outlier handling
outlier_value_changes = data_no_outliers.compare(data_missing)

# Displaying differences
print("\nChanges After Handling Missing Values:")
print(missing_value_changes)

print("\nChanges After Handling Outliers:")
print(outlier_value_changes)

# Step 6: Visualizing the effect of outlier treatment

plt.figure(figsize=(12, 6))

# Before outlier handling
plt.subplot(1, 2, 1)
sns.boxplot(data=data_missing[['Age', 'Salary']])
plt.title('Before Outlier Handling')

# After outlier handling
plt.subplot(1, 2, 2)
sns.boxplot(data=data_no_outliers[['Age', 'Salary']])
plt.title('After Outlier Handling')

plt.tight_layout()
plt.show()
