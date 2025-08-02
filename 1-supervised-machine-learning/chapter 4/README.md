# Chapter 4: Preprocessing Data

This guide provides a comprehensive overview of essential data preprocessing steps required before using machine learning models in `scikit-learn`, with a specific focus on handling categorical features and a practical example using a linear regression model.

## Scikit-learn Data Requirements

For most machine learning algorithms in the `scikit-learn` library, your data must meet the following fundamental requirements:

* **Numeric Data:** All feature columns must be represented by numeric values (e.g., integers, floats). `scikit-learn` models cannot directly process text or other non-numeric data types.
* **No Missing Values:** The dataset should not contain any missing values (e.g., `NaN`, `None`). Missing data must be handled through imputation, deletion, or other strategies before training a model.

In real-world data science projects, it is rare to find a dataset that meets these requirements out-of-the-box. As a result, data preprocessing is a critical and often time-consuming first step.

## Dealing with Categorical Features

Categorical features, which represent distinct groups or labels (e.g., "rock", "pop", "jazz"), must be converted into a numeric format before being fed into a `scikit-learn` model. The most common method for this is **one-hot encoding**.

### One-Hot Encoding

One-hot encoding converts categorical features into a set of binary features, often called **dummy variables**. For each unique category, a new column is created:

* The value is `1` if the observation belongs to that category.
* The value is `0` if the observation does **not** belong to that category.

This process ensures that the model can interpret the categorical information without assuming any ordinal relationship between the categories.

### Tools for Encoding in Python

There are two primary ways to perform one-hot encoding in Python:

* **`pandas`:** The `pandas.get_dummies()` function is a simple and effective way to convert categorical columns into dummy variables, especially during the initial data exploration and preparation phase.
* **`scikit-learn`:** The `sklearn.preprocessing.OneHotEncoder()` class is the recommended approach for integrating one-hot encoding into a machine learning pipeline, as it can be easily included in a `Pipeline` object to ensure consistent preprocessing on both training and test data.

## Example: One-Hot Encoding with `pandas`

This example demonstrates how to use `pandas.get_dummies()` to preprocess a categorical feature before training a linear regression model.

First, let's load a sample dataset containing a `genre` column, which is a categorical feature.

```python
import pandas as pd

# Load the dataset
music_df = pd.read_csv('music.csv')

# Display the first few rows to see the original data
print(music_df.head())
```

### Encoding the Dummy Variables

We will use `pd.get_dummies()` to convert the `genre` column into dummy variables. The `drop_first=True` argument is used to avoid multicollinearity, which is a common issue in linear regression. This drops one of the dummy variable columns, as its information is redundant (e.g., if a row is not 'Rock' and not 'Pop', it must be 'Jazz').

```python
# Create dummy variables for the 'genre' column
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)

# Display the created dummy variables
print(music_dummies.head())
```

### Combining Data and Dropping the Original Column

Next, we combine the new dummy variable columns with the original DataFrame and then drop the original `genre` column.

```python
# Concatenate the original DataFrame with the new dummy variables
music_dummies = pd.concat([music_df, music_dummies], axis=1)

# Drop the original 'genre' column
music_dummies = music_dummies.drop("genre", axis=1)

# Display the preprocessed DataFrame
print(music_dummies.head())
```

## Linear Regression with Dummy Variables

Now that our data is fully numeric, we can use it to train a linear regression model. We will use cross-validation to get a more robust estimate of the model's performance.

```python
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression

# Separate features (X) and target variable (y)
# .values is used to convert the pandas DataFrame/Series to a numpy array, which scikit-learn expects
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up K-fold cross-validation
# n_splits=5 means we will train and test the model 5 times on different data subsets
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Linear Regression model
linreg = LinearRegression()

# Perform cross-validation
# scoring="neg_mean_squared_error" is a common metric; the negative sign is because scikit-learn
# expects a score to be maximized, but MSE is a loss function that we want to minimize.
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")

# Convert the negative MSE scores back to positive, then take the square root to get the RMSE
rmse_scores = np.sqrt(-linreg_cv)

# Print the Root Mean Squared Error (RMSE) for each fold
print("RMSE scores for each fold:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())
```