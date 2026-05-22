import numpy as np

from irest.functions import DiscreteTorus
from irest.transforms import flft, trunc_Z


def etfe(u, y):
    """Empirical Transfer Function Estimate.

    Computes the impulse response estimate using frequency-domain division.

    The method solves the deconvolution problem y = h * u by transforming to the frequency domain,
    performing element-wise division H = Y / U, and transforming back to obtain h.

    Supports both real and complex-valued signals. Automatically selects the appropriate FFT variant
    (rfft/irfft for real, fft/ifft for complex) for efficiency.

    Parameters
    ----------
    u : array_like
        Input signal (length N). Can be real or complex.
    y : array_like
        Output signal (length >= N). Can be real or complex.

    Returns
    -------
    ndarray
        Estimated impulse response (length N). Return type matches input type (real if u and y are
        real, complex otherwise).
    """
    N = len(u)
    u_pad = np.zeros_like(y)
    u_pad[:N] = u

    # Check if inputs are real-valued (all imaginary parts are zero)
    if np.isrealobj(u) and np.isrealobj(y):
        # Use rfft/irfft for real inputs (more efficient)
        return np.fft.irfft(np.fft.rfft(y) / np.fft.rfft(u_pad))[:N]
    else:
        # Use fft/ifft for complex inputs
        return np.fft.ifft(np.fft.fft(y) / np.fft.fft(u_pad))[:N]


def compute_laguerre_fourier_coefficients(u, y, a, fast=True):
    """Compute Laguerre-Fourier coefficients from input and output signals.

    Computes the Laguerre-Fourier coefficients of the transfer function from input signal u
    and output signal y. This is the first step in the Laguerre-Fourier ETFE method.

    - When fast=False (default): Algorithm 2 from Section 3.2. Uses 3 FFT calls via the circulant
      matrix structure. More numerically stable in some cases.
    - When fast=True: Algorithm 3 (Efficient Laguerre-Fourier transform) from Section 3.3. Uses a single
      FFT call for efficiency.

    Both are generalizations of the classical ETFE method (Algorithm 1) that address its limitations
    with band-limited and vanishing-frequency inputs. The method works by transforming the
    deconvolution problem into the Laguerre-Fourier domain.

    Parameters
    ----------
    u : array_like
        Excitation signal in the time domain (length M). Can be real or complex.
    y : array_like
        Measured response signal in the time domain (length N >= M).
    a : scalar or array_like
        Laguerre parameter(s) in the unit disk (|a| < 1). Can be real or complex.
    fast : bool, optional
        If True, uses the efficient single-FFT Algorithm 3. If False (default),
        uses Algorithm 2 with three FFT calls. Default is True.

    Returns
    -------
    ndarray
        Discrete Laguerre-Fourier coefficients of the estimated transfer function
        (length N).
    """
    a = np.asarray(a, dtype=complex)

    N = len(u)
    u_pad = np.zeros_like(y)
    u_pad[:N] = u

    T = DiscreteTorus(a, N)
    u_z, y_z = trunc_Z(u_pad, T), trunc_Z(y, T)

    if fast:
        omega = np.exp(2j * np.pi * np.arange(N) / N)
        u_z *= 1 + a.conj() * omega
        H = y_z / u_z
        return np.fft.fft(H, norm="backward") * np.sqrt(1 - np.abs(a) ** 2) / N
    else:
        u_hat, y_hat = flft(u_z, T), flft(y_z, T)

        gamma = u_hat + a.conj() * np.roll(u_hat, 1)
        gamma /= np.sqrt(1 - np.abs(a) ** 2)

        return etfe(gamma, y_hat)


def laguerre_etfe(u, y, a, fast=False):
    """Laguerre-Fourier Empirical Transfer Function Estimate.

    Computes the impulse response estimate using Laguerre-Fourier expansions.
    This is the complete ETFE algorithm that:
    1. Computes Laguerre-Fourier coefficients from input/output signals
    2. Recovers the impulse response from those coefficients

    - When fast=False (default): Uses Algorithm 2 from Section 3.2 for coefficient computation
    - When fast=True: Uses Algorithm 3 (Efficient) from Section 3.3 for coefficient computation

    Both are generalizations of the classical ETFE method (Algorithm 1) that address its limitations
    with band-limited and vanishing-frequency inputs.

    Parameters
    ----------
    u : array_like
        Excitation signal in the time domain (length M). Can be real or complex.
    y : array_like
        Measured response signal in the time domain (length N >= M).
    a : scalar or array_like
        Laguerre parameter(s) in the unit disk (|a| < 1). Can be real or complex.
    fast : bool, optional
        If True, uses the efficient single-FFT Algorithm 3 for coefficient computation.
        If False (default), uses Algorithm 2 with three FFT calls. Default is False.

    Returns
    -------
    ndarray
        Estimated impulse response (length N).
    """
    # Step 1: Compute Laguerre-Fourier coefficients
    h_lag = compute_laguerre_fourier_coefficients(u, y, a, fast=fast)
    # Step 2: Recover impulse response from coefficients
    N = len(u)
    return recover_ir(h_lag, a, N)


def recover_ir(h_lag, a, N):
    """Recover impulse response from Laguerre-Fourier coefficients.

    Implements the IR recovery algorithm from thm:irrec (Section 3.4 of the paper).
    This function reconstructs the impulse response h from its Laguerre-Fourier
    coefficients h_lag using the inverse transformation formula.

    Parameters
    ----------
    h_lag : array_like
        Laguerre-Fourier coefficients of the impulse response (length N).
    a : scalar
        Laguerre parameter in the unit disk (|a| < 1). Can be real or complex.
    N : int
        Length of the impulse response to recover.

    Returns
    -------
    ndarray
        Recovered impulse response (length N).
    """
    a = np.asarray(a, dtype=complex)
    T = DiscreteTorus(a, N)
    H_hat = np.fft.ifft(h_lag) * N
    fac = (1 - a.conj() * T.z) / np.sqrt(1 - np.abs(a) ** 2)
    h = np.zeros(N, dtype=complex)
    for n in range(N):
        phi_n = fac * T.z**n
        h[n] = np.trapz(H_hat * phi_n.conj()) / N

    return h
