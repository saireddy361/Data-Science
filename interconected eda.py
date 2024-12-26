import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Step 1: Create datasets
# Dataset 1: Employee Performance
data1 = {
    "Employee_ID": [1, 2, 3, 4, 5],
    "Age": [25, 30, 35, 28, 40],
    "Experience": [2, 5, 10, 4, 15],
    "Education_Level": ["Bachelor's", "Master's", "PhD", "Master's", "PhD"],
    "Performance_Score": [75, 85, 95, 80, 90]
}
df1 = pd.DataFrame(data1)

# Dataset 2: Employee Retention
data2 = {
    "Employee_ID": [1, 2, 3, 4, 5],
    "Performance_Score": [75, 85, 95, 80, 90],
    "Job_Satisfaction": [3, 4, 5, 3, 4],
    "Work_Life_Balance": [4, 5, 5, 3, 4],
    "Retention": ["Yes", "Yes", "Yes", "No", "Yes"]
}
df2 = pd.DataFrame(data2)

# Dataset 3: Employee Salary
data3 = {
    "Employee_ID": [1, 2, 3, 4, 5],
    "Experience": [2, 5, 10, 4, 15],
    "Education_Level": ["Bachelor's", "Master's", "PhD", "Master's", "PhD"],
    "Salary": [30000, 50000, 80000, 45000, 100000]
}
df3 = pd.DataFrame(data3)

# Dataset 4: Department Analysis
data4 = {
    "Department_ID": ["D1", "D2", "D3", "D4", "D5"],
    "Performance_Score_Avg": [80, 85, 75, 90, 88],
    "Retention_Rate": [90, 95, 85, 98, 92],
    "Avg_Salary": [45000, 50000, 40000, 60000, 55000]
}
df4 = pd.DataFrame(data4)

# Step 2: Analyze individual datasets
# Dataset 1: Employee Performance
print("Dataset 1 Analysis")
print(df1.describe())

# Exclude non-numeric columns for correlation
numeric_df1 = df1.select_dtypes(include=[np.number])

# Correlation Heatmap
sns.heatmap(numeric_df1.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Employee Performance")
plt.show()

sns.pairplot(df1, hue="Education_Level", palette="viridis")
plt.show()

# Dataset 2: Employee Retention
print("Dataset 2 Analysis")
print(df2.describe())
sns.countplot(x="Retention", data=df2, palette="pastel")
plt.title("Retention Count")
plt.show()

sns.boxplot(x="Retention", y="Performance_Score", data=df2, palette="Set2")
plt.title("Performance Score vs Retention")
plt.show()

# Dataset 3: Employee Salary
print("Dataset 3 Analysis")
print(df3.describe())
sns.histplot(df3["Salary"], kde=True, bins=10, color="skyblue")
plt.title("Salary Distribution")
plt.show()

sns.scatterplot(x="Experience", y="Salary", hue="Education_Level", data=df3, palette="cool")
plt.title("Experience vs Salary")
plt.show()

# Dataset 4: Department Analysis
print("Dataset 4 Analysis")
print(df4.describe())
sns.lineplot(x="Performance_Score_Avg", y="Retention_Rate", marker="o", data=df4, color="red")
plt.title("Performance Score Avg vs Retention Rate")
plt.show()

sns.barplot(x="Department_ID", y="Avg_Salary", data=df4, palette="magma")
plt.title("Average Salary by Department")
plt.show()

# Step 3: Interconnected Analysis
# Merge datasets
merged_df = pd.merge(df1, df2, on="Employee_ID", suffixes=("_Performance", "_Retention"))
merged_df = pd.merge(merged_df, df3, on=["Employee_ID", "Experience", "Education_Level"])

print("Merged Dataset Analysis")
print(merged_df.head())

# Visualize interconnected relationships
sns.scatterplot(
    x="Performance_Score_Performance",  # Adjusted column name
    y="Salary",
    hue="Retention",
    data=merged_df,
    palette="Set1"
)
plt.title("Performance Score vs Salary by Retention")
plt.show()

sns.barplot(
    x="Job_Satisfaction",
    y="Salary",
    hue="Retention",
    data=merged_df,
    palette="cool"
)
plt.title("Job Satisfaction vs Salary by Retention")
plt.show()

# Summary Insights
print("Insights:")
print("""
1. Higher performance scores are linked with higher salaries.
2. Employees with higher job satisfaction are more likely to stay (Retention).
3. Departments with higher average performance scores have better retention rates.
4. Experience and education level significantly affect salary distributions.
""")
