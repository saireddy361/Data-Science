import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Area Safety Prediction.csv')

sns.histplot(data['crimes'], bins=20, kde=True)
plt.title("Distribution of Crimes")
plt.xlabel("Number of Crimes")
plt.ylabel("Frequency")
plt.show()

sns.scatterplot(data=data, x='sex ratio', y='crimes', hue='class')
plt.title("Sex Ratio vs Crimes")
plt.xlabel("Sex Ratio")
plt.ylabel("Number of Crimes")
plt.show()

correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

