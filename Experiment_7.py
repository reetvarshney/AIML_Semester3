import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Creating a sample dataset
data = pd.read_csv('data_100.csv')

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define the features (including 'Name' as it's used in the transformer)
X = df[['Age', 'Salary', 'Department', 'Name']]

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Salary']),
        ('cat', OneHotEncoder(), ['Name', 'Department'])
    ]
)

# Create a pipeline that includes preprocessing
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
X_processed = pipeline.fit_transform(X)

# Get the feature names for the processed DataFrame
num_feature_names = pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Age', 'Salary'])
cat_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['Name', 'Department'])
column_names = np.concatenate([num_feature_names, cat_feature_names])

# Convert the result back to a DataFrame
df_processed = pd.DataFrame(X_processed, columns=column_names)

print("\nProcessed DataFrame:")
print(df_processed)

# Plotting the results

# Set figure size
plt.figure(figsize=(100, 50))

# Plot Original Age vs Salary
plt.subplot(1, 2, 1)
plt.scatter(df['Age'], df['Salary'], color='blue')
plt.title('Original Data: Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

# Plot Processed Age vs Salary (Scaled)
plt.subplot(1, 2, 2)
plt.scatter(df_processed['Age'], df_processed['Salary'], color='orange')
plt.title('Processed Data: Age vs Salary (Scaled)')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Salary')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
