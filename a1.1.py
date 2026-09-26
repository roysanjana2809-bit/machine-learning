import numpy as np

# Internal marks of 10 students
marks = np.array([75, 82, 68, 90, 55, 78, 85, 72, 95, 60])

# Calculations
mean = np.mean(marks)
median = np.median(marks)
std = np.std(marks)
maximum = np.max(marks)
minimum = np.min(marks)

# Print results
print("Marks:", marks)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std)
print("Maximum:", maximum)
print("Minimum:", minimum)