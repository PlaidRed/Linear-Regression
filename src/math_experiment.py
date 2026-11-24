import numpy as np

# Artificial dataset
X = np.array([2, 4, 6])
y = np.array([4, 7, 10])

# Model parameters
b0 = 1
b1 = 1.5

# Calculate predictions
y_pred = b0 + b1 * X

# Calculate residuals
residuals = y - y_pred

# Calculate squared residuals
squared_errors = residuals ** 2

# Calculate Sum of Squared Errors
sse = np.sum(squared_errors)

print("Actual values:", y)
print("Predictions:", y_pred)
print("Residuals:", residuals)
print("Squared errors:", squared_errors)
print("SSE:", sse)
