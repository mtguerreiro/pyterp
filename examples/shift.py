import numpy as np
import matplotlib.pyplot as plt

import pyterp

plt.ion()

# --- Model  ---
f = lambda t : np.cos(2 * np.pi * 10 * t)

# --- Interpolation ---
# Samples
x = 1 / 100 * np.arange(20)
y = f(x)

# Interpolation
xq = x + (1 / 100) / 2 + 10 * (1 / 100)
yq = pyterp.interp.shift(x, xq, y)

# Interpolation error
e = f(xq) - yq

# --- Plots ---
plt.figure(figsize=(6, 4))

plt.plot(xq, f(xq), '--o', label=r'$f(x_q)$')
plt.plot(xq, yq, '--^', label=r'$y_q$')
plt.plot(xq, e, label=r'$f(x_q) - y_q$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.legend()

