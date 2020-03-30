import numpy as np
import numba


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


@numba.njit(parallel=True)
def sinc_numba(x, xq, y, yq_dtype):

    dx = x[1] - x[0]
    yq = np.zeros(xq.shape[0], dtype=yq_dtype)
    
    for n in numba.prange(xq.shape[0]):
        o = np.sinc( (xq[n] - x) / dx ).astype(yq_dtype)
        yq[n] = np.dot(y, o)

    return yq
