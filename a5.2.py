import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 2, 6,
                    3, 5, 7, 4, 8, 2, 6, 7, 3, 5],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 50, 58, 90,
                   62, 72, 88, 68, 92, 54, 78, 86, 59, 74],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
             0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

probability = model.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    y_pred = (probability >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("Threshold:", threshold)
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))