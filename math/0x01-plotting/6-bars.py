#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code goes here ------------------>
width = 0.5

people = ['Farrah', 'Fed', 'Felicia']
f = ['apples', 'bananas', 'oranges', 'peaches']
c = ['red', 'yellow', '#ff8000', '#ffe5b4']

pos0 = None
pos1 = fruit[0]
pos2 = fruit[0] + fruit[1]
pos3 = fruit[0] + fruit[1] + fruit[2]

plt.yticks(np.arange(0, 90, 10))
plt.ylim(0, 80)
plt.suptitle('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')

v = plt.bar(people, fruit[0], width=width, color=c[0], label=f[0], bottom=pos0)
x = plt.bar(people, fruit[1], width=width, color=c[1], label=f[1], bottom=pos1)
y = plt.bar(people, fruit[2], width=width, color=c[2], label=f[2], bottom=pos2)
z = plt.bar(people, fruit[3], width=width, color=c[3], label=f[3], bottom=pos3)

plt.legend(handles=[v, x, y, z])

plt.show()
