import pandas as pd
from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes(as_frame=True)

# Convert dataset into a DataFrame
df = diabetes.frame

# Display first observations
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Dataset information
print("\nDataset information:")
print(df.info())

# Statistical summary
print("\nStatistical summary:")
print(df.describe())
