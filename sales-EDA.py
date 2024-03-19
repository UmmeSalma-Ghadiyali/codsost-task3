import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ignore future warnings related to the use of infinite values as NaNs
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
data = pd.read_csv("sales-dataset.csv")

# Replace infinite values with NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with NaN values
data.dropna(inplace=True)

# Visualize the distribution of the target variable 'Sales'
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=20, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationships between features and the target variable using pairplots
sns.pairplot(data)
plt.suptitle('Pairplot of Features vs. Sales', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()