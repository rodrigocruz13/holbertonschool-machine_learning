#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


y_data = np.arange(0, 11) ** 3
# your code goes here ----------------------- >

# 1. x data
x_data = range(0, 11)

# 2. Colors
line_color = "r"  # line color red
plt.figure(figsize=(10, 7)).set_facecolor("white")  # size & background color

# 3. limits
plt.xlim(0, 10)
plt.ylim(-50, 1050)

# 4. font
plt.rcParams.update({'font.size': 16})

# display
plt.plot(x_data, y_data, line_color, linewidth=3)
plt.grid(False)
plt.show()
