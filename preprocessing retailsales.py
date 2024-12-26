# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('retail_sales_dataset.csv')  # Replace with your file path

# Display the first few rows of the dataset
print("Initial Data Snapshot:")
print(df.head())

# Step 1: Handling Missing Values3
# Fill missing values for numeric fields with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Fill missing values for categorical fields with mode
categorical_cols = df.select_dtypes(include=[object]).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Step 2: Encoding Categorical Variables
# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 3: Scaling Numeric Variables
# Apply Min-Max Scaling
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 4: Feature Engineering
# Example: Create a new feature 'Total_Sale' if 'Quantity' and 'UnitPrice' exist
if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
    df['Total_Sale'] = df['Quantity'] * df['UnitPrice']

# Display the first few rows of the processed dataset
print("Processed Data Snapshot:")
print(df.head())

# Save the processed dataset to a new CSV file
df.to_csv('retail_sales_processed.csv', index=False)
