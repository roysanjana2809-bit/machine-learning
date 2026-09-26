import pandas as pd

# Create DataFrame
data = {
    "Student Name": ["Rahul", "Priya", "Aman", "Sneha", "Riya"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [85, 78, 92, 75, 88],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)

# Filter students who scored above 80
result = df[df["Marks"] > 80]

print("Students who scored above 80 marks:")
print(result)