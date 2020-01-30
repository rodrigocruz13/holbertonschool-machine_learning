#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

align = "mid"  # aligment
edgecolor = 'black'  # edgecolor
n_bins = range(0, 110, 10)
plt.xticks(np.arange(0, 100, step = 10))
plt.hist(student_grades, bins = n_bins, edgecolor = edgecolor)

plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
