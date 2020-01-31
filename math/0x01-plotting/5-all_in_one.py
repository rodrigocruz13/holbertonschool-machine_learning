#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

#  General layout
figure = plt.figure()
figure.suptitle("All in One")

#  Graph 1
figure.add_subplot(321)  # rows, columns, position
x = range(0, 11)
c = "r"  # red
plt.plot(y0, color=c)
plt.xlim(0, 10)

#  Graph 2
figure.add_subplot(322)
s = 4  # point size
c = "m"  # mangenta color
marker = 'o'  # marker circle
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")
plt.scatter(x1, y1, s, c, marker)

#  Graph 3
figure.add_subplot(323)
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of C-14")
plt.xlim(0, 28650)
plt.yscale('log')
plt.plot(x2, y2)

#  Graph 4
figure.add_subplot(324)
color1 = "r--"  # red dash
color2 = "g"  # green
graph1, graph2 = plt.plot(x3, y31, color1, x3, y32, color2)

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of C-14")
plt.legend([graph1, graph2], ["C-14", "Ra-226"])
plt.xlim(0, 20000)
plt.ylim(0, 1)

#  Graph 5 (313)
figure.add_subplot(313)
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")

align = "mid"  # aligment
edgecolor = 'black'  # edgecolor
n_bins = range(0, 110, 10)
plt.xticks(np.arange(0, 100, step=10))
plt.hist(student_grades, bins=n_bins, edgecolor=edgecolor)

plt.xlim(0, 100)
plt.ylim(0, 30)

# General adaptation of the graphs
figure.tight_layout()
plt.show()
