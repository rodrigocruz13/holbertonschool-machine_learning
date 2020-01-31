#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

w = .5
pp = ["Farrah", "Fred", "Felicia"]
col = ["red", "yellow", "#ff8000", "#ffe5b4"]
lab = ["Apples", "Bananas", "Oranges", "Peaches"]

b0 = None
b1 = fruit[0]
b2 = fruit[0] + fruit[1]
b3 = fruit[0] + fruit[1] + fruit[2]

plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.ylim(0, 80)

apple_ = plt.bar(pp, fruit[0], width=w, color=col[0], label=lab[0], bottom=b0)
banana = plt.bar(pp, fruit[1], width=w, color=col[1], label=lab[1], bottom=b1)
orange = plt.bar(pp, fruit[2], width=w, color=col[2], label=lab[2], bottom=b2)
peach_ = plt.bar(pp, fruit[3], width=w, color=col[3], label=lab[3], bottom=b3)

plt.legend(handles=[apple_, banana, orange, peach_])  # Mostrar convenciones
plt.show()
