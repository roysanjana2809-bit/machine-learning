import pandas as pd


data = {
    "Age": [22, 25, None, 28, 30],
    "Salary": [25000, 30000, 35000, None, 45000],
    "Department": ["IT", "HR", "IT", None, "Finance"],
    "Years of Experience": [1, 2, 4, 5, None]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)


print("\n--- Missing Values ---")
print(df.isnull().sum())


df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Years of Experience"] = df["Years of Experience"].fillna(
    df["Years of Experience"].median()
)


df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print("\n--- Preprocessed Dataset ---")
print(df)