import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

# Display basic information
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# Scatter plot: BMI vs Target
plt.figure(figsize=(10, 6))

sns.scatterplot(
        data=df,
        x="bmi",
        y="target",
        alpha=0.6
        )
plt.title("BMI vs Disease Progression")
plt.xlabel("BMI (Standardized)")
plt.ylabel("Disease Progression")
plt.grid(True)

plt.show()

# BMI distribution
plt.figure(figsize=(10, 5))

sns.histplot(
        data=df,
        x="bmi",
        bins=30,
        kde=True
        )
plt.title("BMI Distribution")
plt.xlabel("BMI (Standardized)")
plt.ylabel("Frequency")

plt.show()

# Target distribution
plt.figure(figsize=(10, 5))

sns.histplot(
        data=df,
        x="target",
        bins=30,
        kde=True
        )
plt.title("Target Distribution")
plt.xlabel("Disease Progression")
plt.ylabel("Frequency")

plt.show()

# Calculate BMI and target correlation
correlation = df["bmi"].corr(df["target"])

print("\nBMI and Target Correlation:")
print(correlation)

# Corralation and its visualization
correlation_matrix = df.corr()

# Visualize correlations
plt.figure(figsize=(12, 9))

sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
        )
plt.title("Diabetes Dataset Correlation Matrix")
plt.show()
