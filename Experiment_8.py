# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the CSV data
data = pd.read_csv('expanded_flowers.csv')

# Ensure column names match the dataset (corrected based on initial inspection)
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Target: Species

# 2. Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame containing the two principal components and the species label
df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
df_pca['species'] = y

print("Dataset after PCA transformation:")
print(df_pca.head())

# Explained Variance Ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 4. Visualize the PCA result
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
species_names = ['setosa', 'versicolor', 'virginica']  # Ensure species names match the dataset
for i, species_name in enumerate(species_names):
    plt.scatter(df_pca.loc[df_pca['species'] == species_name, 'PC1'],
                df_pca.loc[df_pca['species'] == species_name, 'PC2'],
                label=species_name, color=colors[i])
    
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Flower Dataset')
plt.legend()
plt.grid(True)
plt.show()

# 5. Cumulative Explained Variance
pca_full = PCA(n_components=X.shape[1])  # Keep all components to see the cumulative effect
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

print("\nCumulative explained variance by all principal components:")
print(cumulative_variance)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, X.shape[1] + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.grid(True)
plt.show()
