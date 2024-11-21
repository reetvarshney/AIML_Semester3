import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for seaborn
sns.set(style="whitegrid")

# Generate a sample dataset for demonstration
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'Values1': np.random.normal(0, 1, 100),
    'Values2': np.random.normal(10, 5, 100),
    'Group': np.random.choice(['G1', 'G2'], 100)
})

# Display the first few rows of the dataset
print(data.head())

# 1. Line Plot (Matplotlib)
plt.figure(figsize=(8, 5))
plt.plot(data['Values1'], label='Values1')
plt.plot(data['Values2'], label='Values2', linestyle='--')
plt.title('Line Plot of Values1 and Values2')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Plot (Seaborn)
plt.figure(figsize=(8, 5))
sns.barplot(x='Category', y='Values1', data=data, palette='viridis')
plt.title('Bar Plot of Values1 by Category')
plt.show()

# 3. Histogram (Matplotlib)
plt.figure(figsize=(8, 5))
plt.hist(data['Values1'], bins=15, alpha=0.7, label='Values1', color='blue')
plt.hist(data['Values2'], bins=15, alpha=0.7, label='Values2', color='green')
plt.title('Histogram of Values1 and Values2')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 4. Box Plot (Seaborn)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Category', y='Values1', data=data, palette='Set2')
plt.title('Box Plot of Values1 by Category')
plt.show()

# 5. Pair Plot (Seaborn)
sns.pairplot(data[['Values1', 'Values2']], diag_kind='kde')
plt.suptitle('Pair Plot of Values1 and Values2', y=1.02)
plt.show()

# 6. Heatmap (Seaborn)
# Create a correlation matrix for the dataset
corr_matrix = data[['Values1', 'Values2']].corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# 7. Scatter Plot (Matplotlib)
plt.figure(figsize=(8, 5))
plt.scatter(data['Values1'], data['Values2'], c='blue', alpha=0.5)
plt.title('Scatter Plot of Values1 vs Values2')
plt.xlabel('Values1')
plt.ylabel('Values2')
plt.grid(True)
plt.show()

# 8. Violin Plot (Seaborn)
plt.figure(figsize=(8, 5))
sns.violinplot(x='Group', y='Values1', data=data, palette='muted')
plt.title('Violin Plot of Values1 by Group')
plt.show()

# 9. Count Plot (Seaborn)
plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=data, palette='Blues')
plt.title('Count Plot of Categories')
plt.show()

# 10. Swarm Plot (Seaborn)
plt.figure(figsize=(8, 5))
sns.swarmplot(x='Category', y='Values1', data=data, palette='Dark2', dodge=True)
plt.title('Swarm Plot of Values1 by Category')
plt.show()