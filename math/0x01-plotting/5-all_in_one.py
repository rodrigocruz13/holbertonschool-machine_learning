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

# your code goes here ---------------------->

#  General layout
fontsize_main = 12
fontsize_title = 'x-small'
fontsize_reg = 'x-small'

figure = plt.figure(figsize=(10, 7))
figure.suptitle("All in One", fontsize=fontsize_main)
figure.set_facecolor("white")   # background color

#  Graph 1
figure.add_subplot(321)  # rows, columns, position
x = range(0, 11)
line_color = "r"  # red
plt.xlim(0, 10)
plt.ylim(-50, 1050)
plt.plot(y0, color=line_color, linewidth=2)
plt.yticks(np.arange(0, 1001, step=500))
plt.xlim(0, 10)


#  Graph 2 "Men's Height vs Weight"
figure.add_subplot(322)
plt.xlabel("Height (in)", fontsize=fontsize_reg)
plt.ylabel("Weight (lbs)", fontsize=fontsize_reg)
plt.title("Men's Height vs Weight", fontsize=fontsize_title)
plt.xlim(54, 83)
plt.ylim(165, 194)
plt.xticks(np.arange(60, 90, step=10))
plt.yticks(np.arange(170, 200, step=10))

volume = 22  # point size
plt.scatter(x1, y1, c='magenta', s=volume, edgecolors='magenta')

#  Graph 3 "Exponential Decay of C-14"
figure.add_subplot(323)
plt.xlabel("Time (years)", fontsize=fontsize_reg)
plt.ylabel("Fraction Remaining", fontsize=fontsize_reg)
plt.title("Exponential Decay of C-14", fontsize=fontsize_title)
plt.xlim(0, 28650)
plt.ylim(0.025, 1.25)
plt.xticks(np.arange(0, 28650, step=10000))
plt.yscale('log')
plt.plot(x2, y2, c='steelblue', linewidth=2)

#  Graph 4 "Exponential Decay of Radioactive Elements"
figure.add_subplot(324)
color1 = "r--"  # red dash
color2 = "g"  # green
graph1, graph2 = plt.plot(x3, y31, color1, x3, y32, color2, linewidth=2.5)
plt.xlabel("Time (years)", fontsize=fontsize_reg)
plt.ylabel("Fraction Remaining", fontsize=fontsize_reg)
plt.title("Exponential Decay of Radioactive Elements", fontsize=fontsize_title)
legend = plt.legend([graph1, graph2], ["C-14", "Ra-226"])
legend.get_frame().set_edgecolor('silver')
plt.yticks(np.arange(0, 1.01, step=0.5))
plt.xticks(np.arange(0, 20001, step=5000))
plt.xlim(0, 20000)
plt.ylim(0, 1)

#  Graph 5 (313) "Project A"
figure.add_subplot(313)
plt.xlabel("Grades", fontsize=fontsize_reg)
plt.ylabel("Number of Students", fontsize=fontsize_reg)
plt.title("Project A", fontsize=fontsize_title)
align = "mid"  # aligment
edgecolor = 'black'  # edgecolor
n_bins = range(0, 110, 10)
plt.xticks(np.arange(0, 101, step=10))
plt.yticks(np.arange(0, 31, step=10))
plt.hist(student_grades, bins=n_bins, edgecolor=edgecolor, color='steelblue')

plt.xlim(0, 100)
plt.ylim(0, 30)

# General adaptation of the graphs
figure.tight_layout()
plt.show()
