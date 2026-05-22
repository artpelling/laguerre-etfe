"""Tests for transforms module (flft, iflft, trunc_Z).

Tests for the Laguerre-Fourier transform functions from the paper
"Impulse Response Estimation with Laguerre Functions".
"""

import numpy as np
import numpy.testing as npt

from irest.functions import DiscreteTorus
from irest.transforms import flft, iflft


def test_flft_iflft_roundtrip(a_param, N_param):
    """Test that flft and iflft are inverses of each other for various a and N.

    Verifies the fast Laguerre-Fourier transform and its inverse correctly
    reconstruct the original signal. See Lemma 2.4 (ii) and (iii) in the paper.
    """
    T = DiscreteTorus(a_param, N_param)

    # Generate a random complex signal
    x = np.random.randn(N_param) + 1j * np.random.randn(N_param)

    # Apply forward and inverse transforms
    x_flft = flft(x, T)
    x_reconstructed = iflft(x_flft, T)

    # Check that we recover the original signal within numerical precision
    # Use rtol=1e-13 to account for floating point errors in complex arithmetic
    npt.assert_allclose(x_reconstructed, x, rtol=1e-13, atol=1e-13)


def test_iflft_flft_roundtrip(a_param, N_param):
    """Test that iflft and flft are inverses (reverse order) for various a and N.

    Verifies the roundtrip property in the reverse direction: applying inverse
    then forward transform should recover the original coefficients.
    """
    T = DiscreteTorus(a_param, N_param)

    # Generate a random complex signal in the transform domain
    x_coeffs = np.random.randn(N_param) + 1j * np.random.randn(N_param)

    # Apply inverse then forward
    x_iflft = iflft(x_coeffs, T)
    x_reconstructed = flft(x_iflft, T)

    # Check reconstruction
    # Use rtol=1e-13 to account for floating point errors in complex arithmetic
    npt.assert_allclose(x_reconstructed, x_coeffs, rtol=1e-13, atol=1e-13)
