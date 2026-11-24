import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
def predict(X, b0, b1):
    """
    Calculate predictions using linear regression.

    X : Input features
    b0 : Intercept
    b1 : Slope
    """

    return b0 + b1 * X

def mean_squared_error(y_actual, y_predicted):
    """
    Calculate Mean Squared Error.
    """

    errors = y_actual - y_predicted

    squared_errors = errors ** 2

    mse = np.mean(squared_errors)

    return mse

# Artificial dataset
X = np.array([1, 2, 3])
y = np.array([2, 3, 5])

# Model parameters
b0 = 1
b1 = 1.2

# Generate predictions
predictions = predict(X, b0, b1)

# Calculate cost
cost = mean_squared_error(y, predictions)

print("Actual values:", y)
print("Predictions:", predictions)
print("MSE:", cost)

# Testing a different parameter combination
experiments = [
        (1, 1.2),
        (0, 1.0),
        (1, 1.5),
        (0, 2.0)
        ]
for b0, b1 in experiments:
    predictions = predict(X, b0, b1)
    cost = mean_squared_error(y, predictions)
    print(f"b0: {b0}, b1: {b1}, MSE: {cost:.4f}")

# Generate possible parameter values
b0_values = np.linspace(-2, 4, 50)
b1_values = np.linspace(0, 3, 50)

# Store calculated costs
costs = np.zeros((len(b0_values), len(b1_values)))

# Calculate MSE for every combination
for i, b0 in enumerate(b0_values):
    for j, b1 in enumerate(b1_values):
        predictions = predict(X, b0, b1)
        costs[i, j] = mean_squared_error(y, predictions)

# Visualize cost function
plt.figure(figsize=(10, 6))

plt.contourf(
        b1_values,
        b0_values,
        costs,
        levels=30,
        cmap="viridis"
        )
plt.colorbar(label="Mean Squared Error")
plt.xlabel("Slope (b1)")
plt.ylabel("Intercept (b0)")
plt.title("Linear Regression Cost Function")
plt.show()

# Load Diabetes dataset
diabetes = load_diabetes(as_frame=True)

df = diabetes.frame

# Extract BMI and target
X_diabetes = df["bmi"].to_numpy()
y_diabetes = df["target"].to_numpy()

# Baseline model: predict the mean target
b0 = np.mean(y_diabetes)
b1 = 0

# Generate predictions
predictions = predict(X_diabetes, b0, b1)

# Calculate MSE
cost = mean_squared_error(y_diabetes, predictions)

print("\nDiabetes Dataset")
print("Intercept:", b0)
print("Slope:", b1)
print("MSE:", cost)
