import numpy as np
from numpy.polynomial import Polynomial


def trunc_Z(x, T):
    """Truncated Z-transformation.

    Computes the truncated Z-transformation as described in the paper.
    See Section 2.2 for the definition of the Discrete Torus and its relationship
    to the Z-transform.

    Parameters
    ----------
    x : array_like
        Input signal.
    T : DiscreteTorus
        Discrete torus object defining the transformation parameters.

    Returns
    -------
    ndarray
        Transformed signal.
    """
    return Polynomial(x)(T.z)


def flft(x, T):
    """Fast Laguerre-Fourier transformation.

    Implements a fast way to compute the discrete Laguerre-Fourier coefficients
    by linear transformation. See Lemma 2.4 (ii) in the paper.

    Parameters
    ----------
    x : array_like
        Input signal.
    T : DiscreteTorus
        Discrete torus object defining the transformation parameters.

    Returns
    -------
    ndarray
        Laguerre-Fourier coefficients of the input signal.
    """
    # x = x.conj()
    xx = np.fft.fft(x / T.sigma**2, norm="ortho", axis=0)
    xx = xx + T.a * np.roll(xx, -1, axis=0)
    fac = np.sqrt(T.N / (1 - np.abs(T.a) ** 2))
    return fac * xx


def iflft(x, T):
    """Fast inverse Laguerre-Fourier transformation.

    Implements a fast way to compute the inverse discrete Laguerre-Fourier
    coefficients by linear transformation. See Lemma 2.4 (iii) in the paper.

    Parameters
    ----------
    x : array_like
        Laguerre-Fourier coefficients.
    T : DiscreteTorus
        Discrete torus object defining the transformation parameters.

    Returns
    -------
    ndarray
        Reconstructed signal in the time domain.
    """
    fac = np.sqrt(T.N / (1 - np.abs(T.a) ** 2))
    xx = x + T.a.conj() * np.roll(x, 1, axis=0)
    return fac * np.fft.ifft(xx, norm="ortho", axis=0)
