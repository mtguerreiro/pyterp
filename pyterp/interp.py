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
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Data dimension
    n = y.shape[0]
    m = y.shape[1]

    if xq.ndim == 1:
        xq = np.tile(xq, (m, 1)).T
    
    # Sampling interval
    dx = x[1] - x[0]

    # The type of the output array will be either float or complex. It cannot
    # be the same type as `y`, since `y` may be of the int type.
    if np.iscomplexobj(y) is True:
        yq_dtype = y.dtype
    else:
        yq_dtype = float
    yq = np.zeros(xq.shape, dtype=yq_dtype)

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

    if yq.shape[1] == 1:
        yq = yq.reshape(-1)
    
    return yq


def sinc(x, xq, y):
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
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Data dimension
    n = y.shape[0]
    m = y.shape[1]

    if xq.ndim == 1:
        xq = np.tile(xq, (m, 1)).T
    
    # Sampling interval
    dx = x[1] - x[0]

    # The type of the output array will be either float or complex. It cannot
    # be the same type as `y`, since `y` may be of the int type.
    if np.iscomplexobj(y) is True:
        yq_dtype = y.dtype
    else:
        yq_dtype = float
    yq = np.zeros(xq.shape, dtype=yq_dtype)
    
    for j in range(y.shape[1]):
        for n, xqn in enumerate(xq[:, j]):
            #yq[n, j] = np.sum(y[:, j] * kernel(xqn - x, kind=kernel))
            yq[n, j] = np.dot(y[:, j], kernel(xqn - x, kind='sinc'))

    if yq.shape[1] == 1:
        yq = yq.reshape(-1)

    return yq


def kernel(x, kind='linear'):
    """Interpolation kernel for interpolation routines.

    Parameters
    ----------
    x : :class:`np.ndarray`
        x-axis for the interpolation kernel.

    kind : :class:`str`
        Interpolation kernel. Can be `linear` or `sinc`.

    Returns
    -------
    h : :class:`np.ndarray`
        Kernel at points `x`.
        
    """
    # Sampling interval
    dx = np.abs(x[1] - x[0])

    # Output
    o = np.zeros(x.shape)

    if kind == 'linear':
        xabs = np.abs(x)
        i = xabs <= dx
        o[i] = 1 - xabs[i] / dx

    elif kind == 'sinc':
        o[:] = np.sinc(x / dx)

    return o
