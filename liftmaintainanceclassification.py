import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data
file_path = 'C:/Users/Sai Baba Reddy/data science/LiftMaintainancePrediction.csv'
df = pd.read_csv(file_path)

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Handle Missing Values
df.ffill(inplace=True)  # Updated to use ffill directly

# Step 4: Encode Categorical Features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Feature Scaling
scaler = StandardScaler()
X = df.drop('CLASS', axis=1)  # Updated to drop 'CLASS' instead of 'target'
y = df['CLASS']
X = scaler.fit_transform(X)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
