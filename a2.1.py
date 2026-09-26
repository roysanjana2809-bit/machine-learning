import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()


df = pd.DataFrame(wine.data, columns=wine.feature_names)


df['target'] = wine.target


print("First 5 rows:")
print(df.head())


print("\nShape:")
print(df.shape)


print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


print("\nWine Classes:")
print(df['target'].value_counts())


df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot")
plt.show()