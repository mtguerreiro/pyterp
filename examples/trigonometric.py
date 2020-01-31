import numpy as np
import matplotlib.pyplot as plt

import pyterp

plt.ion()

# --- Model  ---
f = lambda t : np.sin(10 * t)

# --- Signal ---
x = np.arange(0, 1, 0.01)
y = f(x)

# --- Interpolation ---
# Query points
#xq = np.arange(-3, 3, 0.1)
xq = x

# Direct and inverse DFT
N = x.shape[0]
n = np.arange(N)
k = n.reshape(-1, 1)
W = 1 / np.sqrt(N) * np.exp(-1j * 2 * np.pi * n * k / N)
W_inv = 1 / np.sqrt(N) * np.exp(1j * 2 * np.pi * n * k / N)

# Coefficients
D = W @ y.reshape(-1, 1)

yq = np.zeros(xq.shape, dtype=complex)
for i in range(x.shape[0]):
    yq[i] = 1 / np.sqrt(N) * np.sum(D * np.exp(1j * 2 * np.pi * k * xq[i]))

### Interpolation
##xq = np.arange(-3, 3, 0.1)
##yq = pyterp.interp.linear(x, xq, y)
##
### Interpolation error
##e = f(xq) - yq
##
### --- Plots ---
##plt.figure(figsize=(6, 4))
##
##plt.plot(xq, f(xq), label=r'$f(x_q)$')
##plt.plot(xq, yq, label=r'$y_q$')
##plt.plot(xq, e, label=r'$f(x_q) - y_q$')
##
##plt.xlabel('$x$')
##plt.ylabel('$y$')
##plt.grid()
##plt.legend()

