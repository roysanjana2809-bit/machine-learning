import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create dataset
data = {
    "Age": [20, 21, 22, 23, 24],
    "Salary": [25000, 30000, 35000, 40000, 45000]
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# StandardScaler
standard = StandardScaler()
standard_data = standard.fit_transform(df)

print("\n--- StandardScaler Output ---")
print(standard_data)

# MinMaxScaler
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(df)

print("\n--- MinMaxScaler Output ---")
print(minmax_data)

# Show ranges
print("\nStandardScaler Range:")
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())

print("\nMinMaxScaler Range:")
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())