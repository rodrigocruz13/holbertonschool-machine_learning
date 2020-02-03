#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

color1 = "r--"  # red dash
color2 = "g"  # green

graph1, graph2 = plt.plot(x, y1, color1, x, y2, color2)

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")
plt.legend([graph1, graph2], ["C-14", "Ra-226"])

plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.show()
