import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('sales-dataset.csv')

# Select features and target variable
X = data[['TV', 'Radio', 'Newspaper']].values  # Features (input)
y = data['Sales'].values.reshape(-1, 1)  # Target variable (output)

# Add a column of ones to X for the bias term
X = np.c_[np.ones(X.shape[0]), X]

# Define the linear regression function
def linear_regression(X, y):
    # Compute the coefficients (theta) using the normal equation
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Train the model to get the coefficients
theta = linear_regression(X, y)

# Print the coefficients (theta)
print('Coefficients (theta):')
print(theta)

# Make predictions using the trained model
y_pred = X.dot(theta)

# Calculate the mean squared error (MSE)
mse = np.mean((y_pred - y) ** 2)
print(f'Mean Squared Error (MSE): {mse:.2f}')

# Calculate the R-squared score
y_mean = np.mean(y)
sst = np.sum((y - y_mean) ** 2)
ssr = np.sum((y_pred - y_mean) ** 2)
r_squared = ssr / sst
print(f'R-squared Score: {r_squared:.2f}')

# Create a DataFrame to display actual vs. predicted sales and residuals
results = pd.DataFrame({'Actual Sales': y.flatten(), 'Predicted Sales': y_pred.flatten(),
                        'Residuals': (y_pred - y).flatten()})
print('\nActual vs. Predicted Sales:')
print(results.head(10))  # Display the first 10 rows

# Plotting actual vs. predicted values
plt.scatter(y, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Sales')
plt.show()