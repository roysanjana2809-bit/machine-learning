import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. student dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 2, 6],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 50, 58, 90],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

print("--- Student Dataset ---")
print(df)

# 2. Separate features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
print("\n--- Logistic Regression Performance ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))