import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

data = pd.read_csv("clean_data.csv")

sns.countplot(x = 'Age', data=data) 
plt.title('Age Distribution')
plt.show()

sns.countplot(x = 'Salary', data = data) 
plt.title('salary Distributiion')
plt.show()

sns.countplot(x = 'Department', data = data)
plt.title('Department Distribution')
plt.show()

sns.heatmap(data[['Age', 'Salary']].corr(), annot=True, cmap = 'coolwarm', linewidths = 0.5)
plt.title("Correlation Heatmap")
plt.show()
