#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here ---------------------->

# 0 frame
plt.figure(figsize=(10, 7)).set_facecolor("white")

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of C-14")

# 2. Colors
line_color = '#67c2db'  # line color blue

# 3. font
plt.rcParams.update({'font.size': 16})
plt.yscale('log')
plt.xlim(0, 28650)
plt.ylim(0.025, 1.25)


plt.plot(x, y, c='steelblue', linewidth=3)
plt.show()
