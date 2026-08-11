import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.arange(-5, 5, 0.1)

y = sigmoid(x)

plt.plot(x, y)

plt.title("Visualization of Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")

plt.grid()
plt.show()
