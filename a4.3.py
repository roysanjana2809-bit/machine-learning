import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Load dataset
df = pd.read_csv("housing_multiple.csv")

print("--- First Five Rows ---")
print(df.head())

# 2. Select features and target
X = df[["area_sqft", "bedrooms"]]
y = df["price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Standard Multiple Linear Regression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Multiple Linear Regression ---")
print("R2 Score:", round(r2_linear, 4))


# 5. Polynomial Regression

# Create polynomial features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Prediction
y_pred_poly = poly_model.predict(X_test_poly)

# R2 score
r2_poly = r2_score(y_test, y_pred_poly)

print("\n--- Polynomial Regression ---")
print("R2 Score:", round(r2_poly, 4))



# 6. Compare R2 Scores

print("\n--- R2 Score Comparison ---")
print("Linear Regression   :", round(r2_linear, 4))
print("Polynomial Regression:", round(r2_poly, 4))

if r2_poly > r2_linear:
    print("\nPolynomial Regression has a higher R2 score.")
elif r2_poly < r2_linear:
    print("\nLinear Regression has a higher R2 score.")
else:
    print("\nBoth models have the same R2 score.")