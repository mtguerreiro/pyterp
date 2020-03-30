import numpy as np
import numba
import pyterp


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
        yc = np.ascontiguousarray(y[:, j])
        yq[:, j] = pyterp.kernels.sinc_numba(x, xq[:, j], yc, yq.dtype)

    if yq.shape[1] == 1:
        yq = yq.reshape(-1)

    return yq


def shift(x, xq, y):
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
    nq = xq.shape[0]

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

    # Padding size
    n_s = 4

    # Signal frequencies
    fs = 1 / dx
    f = np.zeros(n)
    f[:int(n / 2)] = fs / n * np.arange(0, int(n / 2))
    f[int(n / 2):] = fs / n * np.arange(-int(n / 2), 0)

    # Padding frequencies
    fx = np.zeros(n_s)
    fx[:int(n_s / 2)] = f[:int(n_s / 2)]
    fx[int(n_s / 2):] = f[-int(n_s / 2):]

    # Padding vector
    s_l = np.zeros(n_s)

    x_min = x.min()
    for j in range(m):
        yqq = np.zeros(nq, dtype=complex)

        for i in range(nq):
            Ii = (np.arange(-int(n_s / 2), int(n_s / 2)) + i) % nq
            # Index of sample on x closest to xq(i, j)
            iq = (xq[i, j] - x_min) / dx
            iq = (iq % n).astype(int)
            # Amount of shift
            delta = xq[i, j] - iq * dx

            s_l[int(n_s / 2)] = y[iq, j]

            S_l = np.fft.fft(s_l)
            S_ll = S_l * np.exp(-1j * 2 * np.pi * fx * -delta)

            s_ll = np.fft.ifft(S_ll)

            yqq[Ii] = yqq[Ii] + s_ll

        yq[:, j] = yqq.real

    if yq.shape[1] == 1:
        yq = yq.reshape(-1)

    return yq
