import numpy as np
import matplotlib.pyplot as plt

import pyterp

plt.ion()

# --- Model  ---
f = lambda t : t ** 3 - t ** 2 + 10

# --- Interpolation ---
# Samples
x = np.arange(-5, 5, 0.5)
y = f(x)

# Interpolation
xq = np.arange(-3, 3, 0.1)
yq = pyterp.interp.linear(x, xq, y)

# Interpolation error
e = f(xq) - yq

# --- Plots ---
plt.figure(figsize=(6, 4))

plt.plot(xq, f(xq), label=r'$f(x_q)$')
plt.plot(xq, yq, label=r'$y_q$')
plt.plot(xq, e, label=r'$f(x_q) - y_q$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()

