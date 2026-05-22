"""Mathematical functions for Laguerre-Fourier analysis.

This module provides the Blaschke factor and Discrete Torus classes that form the mathematical
foundation for the Laguerre-Fourier transforms.
"""

from functools import cached_property

import numpy as np


class Blaschke(object):
    """Blaschke factor for Möbius transformations.

    Blaschke factors are the building blocks of Laguerre functions and are used in the
    discretization of the Laguerre-Fourier transform. They are Möbius transformations that map the
    unit disk to itself.

    A Blaschke factor with parameter a ∈ D (unit disk) is defined as: B_a(z) = (z - a) / (1 - a̅ z)

    Parameters
    ----------
    a : scalar
        Parameter in the unit disk (|a| < 1). Can be real or complex.

    Attributes
    ----------
    a : ndarray
        The Blaschke parameter as a complex numpy array.
    """

    def __init__(self, a):
        """Initialize a Blaschke factor with parameter a.

        Parameters
        ----------
        a : scalar
            Parameter in the unit disk (|a| < 1).
        """
        super().__init__()
        self.a = np.array(a, dtype=complex)

    def __call__(self, z):
        """Evaluate the Blaschke factor at points z.

        Computes B_a(z) = (z - a) / (1 - a̅ z) for each element of z.

        Parameters
        ----------
        z : array_like
            Points at which to evaluate the Blaschke factor. Typically on the unit circle T or in
            the unit disk D.

        Returns
        -------
        ndarray
            The Blaschke factor evaluated at each point in z.
        """
        return (z - self.a) / (1 - self.a.conj() * z)

    @property
    def inverse(self):
        """Inverse Blaschke factor.

        Returns
        -------
        Blaschke
            A new Blaschke instance with parameter -a.
        """
        return Blaschke(-self.a)


class DiscreteTorus(object):
    """Discrete Torus for Laguerre-Fourier sampling.

    The Discrete Torus T_N^a is a discretization of the unit circle T that ensures discrete
    orthogonality of the sampled Laguerre functions. It is defined as the set of points z_k where
    the Blaschke factor raised to power N equals 1.

    Parameters
    ----------
    a : scalar
        Laguerre parameter in the unit disk (|a| < 1). Controls the
        sampling grid on the unit circle.
    N : int
        Number of sampling points. Must be positive.

    Attributes
    ----------
    a : ndarray
        The Laguerre parameter as a complex numpy array.
    N : int
        Number of sampling points.
    z : ndarray
        The sampling points on the unit circle (length N).
    sigma : ndarray
        The discrete measure at each sampling point (length N).
    """

    def __init__(self, a, N):
        """Initialize a Discrete Torus with parameter a and N sampling points.

        Parameters
        ----------
        a : scalar
            Laguerre parameter in the unit disk (|a| < 1).
        N : int
            Number of sampling points. Must be positive.
        """
        super().__init__()
        self.a = np.array(a, dtype=complex)
        N = int(N)
        assert N > 0
        self.N = N

    @cached_property
    def z(self):
        """Sampling points on the unit circle.

        The points z_k are computed as z_k = B_{-a}(ω^k) where ω^k are the N-th roots of unity.

        Returns
        -------
        ndarray
            Complex array of N sampling points on the unit circle.
        """
        return Blaschke(self.a).inverse(np.exp(2j * np.pi * np.arange(self.N) / self.N))

    @cached_property
    def sigma(self):
        """Discrete measure at each sampling point.

        The discrete measure σ(z_k) is given by: σ(z) = sqrt(N(1 - |a|^2)) / |1 - a̅ z|

        This measure ensures discrete orthogonality of the sampled Laguerre functions with respect
        to the inner product.

        Returns
        -------
        ndarray
            Real-valued array of length N containing the discrete measure at each sampling point
            z_k.
        """
        return np.sqrt(self.N * (1 - np.abs(self.a) ** 2)) / np.abs(1 - np.conj(self.a) * self.z)
