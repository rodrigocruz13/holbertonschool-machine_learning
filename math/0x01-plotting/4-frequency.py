#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here ------------------------------>

# 0 frame
plt.figure(figsize=(10, 7)).set_facecolor("white")

# labels
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

align = "mid"  # aligment
edgecolor = 'black'  # edgecolor
innercolor = 'steelblue'

# 3. font
plt.rcParams.update({'font.size': 13.5})

n_bins = range(0, 110, 10)
plt.xticks(np.arange(0, 101, step=10))
plt.hist(student_grades,
         bins=n_bins,
         edgecolor=edgecolor,
         linewidth=1.5,
         color=innercolor)

plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
