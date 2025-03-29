import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data around a predefined line (y = kx + b)
k_true = 2.5  # True slope
b_true = 1.0  # True intercept

# Number of points
n_points = 100

# Generate data with noise
np.random.seed(42)  # For reproducibility
x = np.random.uniform(-10, 10, n_points)  # Uniform distribution for x
noise = np.random.normal(0, 5, n_points)  # Normal noise
y = k_true * x + b_true + noise  # Add noise to the line

print("First 5 points (x, y):")
for i in range(5):
    print(f"x={x[i]:.2f}, y={y[i]:.2f}")


# 2. Least Squares Method
def least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    k = numerator / denominator
    b = y_mean - k * x_mean

    return k, b


# Estimate parameters using least squares
k_est, b_est = least_squares(x, y)
print(f"Least Squares: k = {k_est:.2f}, b = {b_est:.2f}")

# Compare with np.polyfit
coefficients = np.polyfit(x, y, 1)
k_polyfit, b_polyfit = coefficients
print(f"np.polyfit: k = {k_polyfit:.2f}, b = {b_polyfit:.2f}")


# 3. Gradient Descent
def gradient_descent(x, y, learning_rate=0.01, n_iter=1000):
    n = len(x)
    k = 0  # Initial guess for k
    b = 0  # Initial guess for b
    errors = []  # To store MSE values

    for _ in range(n_iter):
        y_pred = k * x + b  # Predicted y values

        # Compute gradients
        dk = -(2 / n) * np.sum(x * (y - y_pred))
        db = -(2 / n) * np.sum(y - y_pred)

        # Update parameters
        k -= learning_rate * dk
        b -= learning_rate * db

        # Compute MSE
        mse = np.mean((y - y_pred) ** 2)

        # Debugging output
        if _ % 100 == 0:  # Print every 100th iteration
            print(f"Iteration {_}: k = {k:.4f}, b = {b:.4f}, MSE = {mse:.4f}")

        # Check for invalid MSE
        if np.isnan(mse) or np.isinf(mse):
            print("MSE is NaN or Inf! Stopping...")
            break

        errors.append(mse)

    return k, b, errors


# Estimate parameters using gradient descent
learning_rate = 0.01
n_iter = 1000
k_gd, b_gd, errors = gradient_descent(x, y, learning_rate, n_iter)
print(f"Gradient Descent: k = {k_gd:.2f}, b = {b_gd:.2f}")

# 4. Visualization
# Plot the data and regression lines
plt.figure(figsize=(10, 6))

# Scatter plot of the data
plt.scatter(x, y, color='blue', label='Data', alpha=0.6)

# True line
x_range = np.linspace(-10, 10, 100)
y_true = k_true * x_range + b_true
plt.plot(x_range, y_true, color='green', label=f'True Line (y = {k_true:.2f}x + {b_true:.2f})', linestyle='--')

# Least Squares line
y_est = k_est * x_range + b_est
plt.plot(x_range, y_est, color='red', label=f'Least Squares (y = {k_est:.2f}x + {b_est:.2f})')

# np.polyfit line
y_polyfit = k_polyfit * x_range + b_polyfit
plt.plot(x_range, y_polyfit, color='purple', label=f'np.polyfit (y = {k_polyfit:.2f}x + {b_polyfit:.2f})',
         linestyle=':')

# Gradient Descent line
y_gd = k_gd * x_range + b_gd
plt.plot(x_range, y_gd, color='orange', label=f'Gradient Descent (y = {k_gd:.2f}x + {b_gd:.2f})', linestyle='-.')

# Plot settings
plt.title('Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Plot the error (MSE) vs iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iter + 1), errors, color='teal')
plt.title('Error (MSE) vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(len(errors)), errors, color='teal')

# Zoom into the first 50 iterations
plt.xlim(left=0, right=49)
plt.ylim(bottom=0, top=(errors[0]) * 1.1)

# Plot settings
plt.title('Error (MSE) vs Iterations (First 50 Iterations)')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

normalized_errors = (errors - min(errors)) / (max(errors) - min(errors))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(normalized_errors) + 1), normalized_errors, color='teal')

# Plot settings
plt.title('Normalized Error (MSE) vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Normalized MSE')
plt.grid(True)
plt.show()

