import pandas as pd

# Create DataFrame
data = {
    "Student Name": ["Rahul", "Priya", "Aman", "Sneha", "Riya"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [95, 85, 72, 68, 90],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)

# Add Grade column
def grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"

df["Grade"] = df["Marks"].apply(grade)

print(df)