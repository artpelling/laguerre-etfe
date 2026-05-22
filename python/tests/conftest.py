"""Pytest configuration and shared fixtures for irest tests."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set a consistent random seed for all tests."""
    np.random.seed(42)


@pytest.fixture(
    params=[
        # Real values: small, medium, large (but < 1)
        0.0,
        0.1,
        0.5,
        0.9,
        # Pure imaginary values
        0.1j,
        0.5j,
        0.9j,
        # General complex values in unit disk
        0.3 + 0.4j,
        0.1 + 0.1j,
        0.5 + 0.5j,
        0.7 + 0.2j,
    ]
)
def a_param(request):
    """Fixture providing various a parameters in the interior of the unit disk.

    Shared across test modules. The parameter a is the Laguerre parameter
    used in the Discrete Torus and Laguerre-Fourier transforms.
    """
    return request.param


@pytest.fixture(params=[8, 16, 32])
def N_param(request):
    """Fixture providing various signal lengths.

    Shared across test modules. Used for testing transforms with different
    signal sizes.
    """
    return request.param


@pytest.fixture
def N_fixed():
    """Fixed signal length for tests that need consistent N."""
    return 32


@pytest.fixture
def N_small():
    """Small signal length for quick tests."""
    return 16
