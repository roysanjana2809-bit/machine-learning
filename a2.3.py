import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


wine = load_wine()


df = pd.DataFrame(wine.data, columns=wine.feature_names)


corr = df.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


corr_pairs = corr.where(
    ~pd.np.tril(pd.np.ones(corr.shape)).astype(bool)
)

print(corr_pairs.stack().sort_values(ascending=False).head())