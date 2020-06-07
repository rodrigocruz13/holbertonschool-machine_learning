#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# your code here ----------------------->

# 1. Marker size in units of points ^ 2
volume = (6)**2
plt.figure(figsize=(12, 8)).set_facecolor("white")

# 2. limits
plt.xlim(54, 84)
plt.ylim(165, 194)

# 3. font
plt.rcParams.update({'font.size': 16})

# 4.labels
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.suptitle("Men's Height vs Weight")

# 5. Display
plt.scatter(x, y, c='magenta', s=volume, edgecolors='magenta')
plt.show()
