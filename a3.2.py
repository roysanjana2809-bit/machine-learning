import pandas as pd
from sklearn.preprocessing import MinMaxScaler


data = {
    "Age": [20, 21, 22, 23, 24],
    "Salary": [25000, 30000, 35000, 40000, 45000]
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# Apply MinMaxScaler
scaler = MinMaxScaler()

df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

print("\n--- Scaled Data ---")
print(df)