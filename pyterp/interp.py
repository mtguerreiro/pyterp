import numpy as np


def linear(x, xq, y):
    """Interpolates each column of a column-matrix :math:`x_q` on a matrix
    :math:`y`.

    Parameters
    ----------
    x : :class:`np.ndarray`
        Points where :math:`y` is known. Must be a vector of size (n_x,).

    xq : :class:`np.ndarray`
        Values to interpolate. Must be a vector of size (n_xq,) or a matrix of
        size (n_xq, n_y).

    y : :class:`np.ndarray`
        Known values of :math:`y`. Must be a vector of size (n_x, n_y).

    Returns
    -------
    yq : :class:`np.ndarray`
        Interpolated values.
        
    """
    
    # Data dimension
    n = y.shape[0]
    m = y.shape[1]

    if xq.ndim == 1:
        xq = np.tile(xq, (m, 1)).T
    
    # Sampling interval
    dx = x[1] - x[0]

    yq = np.zeros(xq.shape, dtype=y.dtype)

    for j in range(y.shape[1]):

        # Computes indexes of query points
        x_idx = np.floor((xq[:, j] - x[0]) / dx).astype(int)
        idx = np.logical_and(x_idx > 0, x_idx < (n - 2))
        np.clip(x_idx, 0, n - 2, out=x_idx)

        # Angular coefficient of each interpolation point
        m = (y[x_idx + 1, j] - y[x_idx, j]) / (x[x_idx + 1] - x[x_idx])
        
        # Interpolation
        yq[:, j] = y[x_idx, j] + m * (xq[:, j] - x[x_idx])
        yq[~idx] = 0

    return yq
