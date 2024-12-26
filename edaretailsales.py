import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('retail_sales_processed.csv')

# Univariate Analysis: Histogram for Unit Price
sns.histplot(df['UnitPrice'], bins=20, kde=True)
plt.title("Distribution of Unit Price")
plt.xlabel("Unit Price")
plt.ylabel("Frequency")
plt.show()

# Bivariate Analysis: Scatter Plot for Quantity vs Unit Price
sns.scatterplot(data=df, x='Quantity', y='UnitPrice', hue='Total_Sale')
plt.title("Quantity vs Unit Price")
plt.xlabel("Quantity")
plt.ylabel("Unit Price")
plt.show()

# Multivariate Analysis: Heatmap of Correlation Matrix
correlation = df[['Quantity', 'UnitPrice', 'Total_Sale']].corr()  # Focus on key columns
plt.figure(figsize=(8, 6))  # Adjust figure size
sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f')  # Subtle color, rounded values
plt.title("Correlation Heatmap")
plt.show()
