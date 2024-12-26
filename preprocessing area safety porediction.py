import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('Area Safety Prediction.csv')

# Display initial data snapshot
print("Initial Data Snapshot:")
print(data.head())

# Step 1: Handling Missing Values
# Fill missing numeric values with median
numeric_cols = data.select_dtypes(include=[np.number]).columns
if data[numeric_cols].isnull().sum().sum() > 0:
    data[numeric_cols] = data[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Fill missing categorical values with mode
categorical_cols = data.select_dtypes(include=[object]).columns
if data[categorical_cols].isnull().sum().sum() > 0:
    data[categorical_cols] = data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Step 2: Encoding Categorical Variables
# Identify categorical columns if any and one-hot encode them
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Step 3: Scaling Numeric Variables
# Apply Min-Max Scaling to numeric features
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Display the processed data snapshot
print("Processed Data Snapshot:")
print(data.head())

# Step 4: Save Processed Data
# Export to a new CSV file
data.to_csv('Area_Safety_Processed.csv', index=False)
print("Processed data saved as 'Area_Safety_Processed.csv'.")
