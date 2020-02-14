import numpy as np
import matplotlib.pyplot as plt

import pyterp

plt.ion()

# --- Model  ---
f = lambda t : t ** 3 - t ** 2 + 10
#f = lambda t : np.sin(t)

# --- Interpolation ---
# Samples
x = np.arange(-5, 5, 0.03)
y = f(x)

# Interpolation
xq = np.arange(-10, 10, 0.1)
yq = pyterp.interp.sinc(x, xq, y)

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

