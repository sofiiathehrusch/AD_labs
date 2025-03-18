import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Fetch the dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Check for missing values
print("Missing values in X:", X.isna().sum())
print("Missing values in y:", y.isna().sum())

# Handle missing values by replacing with median
X = X.fillna(X.median())
print("Missing values in X after handling:", X.isna().sum())

# Convert to numpy array for further processing
X_np = X.to_numpy()

# Define normalization and standardization functions
def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def standardize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Normalize and standardize numeric columns
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X_normalized = normalize(X[numeric_columns].to_numpy())
X_standardized = standardize(X[numeric_columns].to_numpy())

# Print normalized and standardized data
print("Normalized Data:\n", X_normalized[:5])
print("Standardized Data:\n", X_standardized[:5])

# Build histogram
sns.histplot(X['age'], bins=10, kde=True, color='green')
plt.title('Histogram of Age with KDE')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Build scatter plot
sns.scatterplot(data=X, x='age', y='chol', color='purple')
plt.title('Scatter Plot of Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# Calculate Pearson and Spearman correlations
pearson_corr, _ = pearsonr(X['age'], X['chol'])
spearman_corr, _ = spearmanr(X['age'], X['chol'])
print(f"Pearson Correlation Coefficient: {pearson_corr}")
print(f"Spearman Correlation Coefficient: {spearman_corr}")

# One Hot Encoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_encoded = encoder.fit_transform(X[categorical_columns])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    X_final = pd.concat([X.drop(columns=categorical_columns), X_encoded_df], axis=1)
else:
    X_final = X

# Combine X_final and y into a single DataFrame
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Extract the first column as a Series
X_final_with_target = X_final.copy()
X_final_with_target['target'] = y  # Add y as a new column

# Pairplot visualization
sns.pairplot(X_final_with_target, hue='target')  # Use 'target' for hue
plt.show()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.5, random_state=42
)
print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

# Initialize regression models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)  # Regularization strength
lasso_reg = Lasso(alpha=0.1)  # Regularization strength

# Train the models
linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)

# Function to plot true vs predicted values
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')  # Ideal line
    plt.title(title)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

# Function to plot residuals
def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='green')
    plt.axhline(y=0, color='red', linestyle='--')  # Zero residual line
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

# Visualize predictions and residuals for each model
plot_predictions(y_test, y_pred_linear, "Linear Regression Predictions")
plot_residuals(y_test, y_pred_linear, "Linear Regression Residuals")

plot_predictions(y_test, y_pred_ridge, "Ridge Regression Predictions")
plot_residuals(y_test, y_pred_ridge, "Ridge Regression Residuals")

plot_predictions(y_test, y_pred_lasso, "Lasso Regression Predictions")
plot_residuals(y_test, y_pred_lasso, "Lasso Regression Residuals")

# Calculate MSE for each model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print the results
print(f"Linear Regression MSE: {mse_linear}")
print(f"Ridge Regression MSE: {mse_ridge}")
print(f"Lasso Regression MSE: {mse_lasso}")

# Determine the best model
best_model = min(
    [("Linear", mse_linear), ("Ridge", mse_ridge), ("Lasso", mse_lasso)],
    key=lambda x: x[1]
)
print(f"Best Model: {best_model[0]} with MSE: {best_model[1]}")

print("Features used for training:", X_train.columns)