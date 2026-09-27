import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("housing_cleaned.csv")

print("--- First Five Rows ---")
print(df.head())

# 2. Select area and price
X = df[["area_sqft"]]
y = df["price"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2 :", round(r2, 4))

print("\nSlope:", model.coef_[0])
print("Intercept:", model.intercept_)

# 7. Plot
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")

plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.title("House Area vs Price")
plt.legend()
plt.show()