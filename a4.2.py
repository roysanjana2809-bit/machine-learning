import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = pd.read_csv("housing_multiple.csv")

print("--- First Five Rows ---")
print(df.head())


X = df[["area_sqft", "bedrooms"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))


print("\n--- Model Coefficients ---")
print("Area coefficient    :", round(model.coef_[0], 2))
print("Bedroom coefficient :", round(model.coef_[1], 2))
print("Intercept            :", round(model.intercept_, 2))


print("\n--- Regression Equation ---")
print(
    "Price =",
    round(model.intercept_, 2),
    "+",
    round(model.coef_[0], 2),
    "* Area",
    "+",
    round(model.coef_[1], 2),
    "* Bedrooms"
)


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()