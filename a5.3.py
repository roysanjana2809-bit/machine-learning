import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Student data
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 2, 6,
                    3, 5, 7, 4, 8, 2, 6, 7, 3, 5],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 50, 58, 90,
                   62, 72, 88, 68, 92, 54, 78, 86, 59, 74],
    "Pass": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
             0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Input and output
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction probability
y_prob = model.predict_proba(X_test)[:, 1]

# ROC and AUC
fpr, tpr, threshold = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("AUC Score:", round(auc, 4))

# Plot ROC curve
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()