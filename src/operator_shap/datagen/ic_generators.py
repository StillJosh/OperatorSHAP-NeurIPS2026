"""Random initial condition generators for PDE data generation.

Each generator takes a spatial grid and a numpy RNG, returning one IC sample.
"""

import numpy as np


def random_fourier_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    n_modes: int = 20,
    power_decay: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Sum of random Fourier modes with power-law amplitude decay.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on [0, L].
    rng : np.random.Generator
        Numpy random generator.
    n_modes : int
        Number of Fourier modes.
    power_decay : float
        Exponent for amplitude decay: a_k ~ k^{-power_decay}.
    amplitude : float
        Overall scale factor.

    Returns
    -------
    np.ndarray
        Initial condition, shape ``(len(x),)``.
    """
    L = x[-1] - x[0] + (x[1] - x[0])  # period length
    u = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        a_k = rng.normal() * amplitude * k ** (-power_decay)
        phi_k = rng.uniform(0, 2 * np.pi)
        u += a_k * np.sin(2 * np.pi * k * x / L + phi_k)
    return u


def gaussian_bumps_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    n_bumps_range: tuple[int, int] = (1, 4),
    amplitude: float = 1.0,
) -> np.ndarray:
    """Sum of random Gaussian bumps.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on [0, L].
    rng : np.random.Generator
        Numpy random generator.
    n_bumps_range : tuple[int, int]
        Range for the number of bumps (inclusive).
    amplitude : float
        Overall scale factor.

    Returns
    -------
    np.ndarray
        Initial condition, shape ``(len(x),)``.
    """
    L = x[-1] - x[0]
    n_bumps = rng.integers(n_bumps_range[0], n_bumps_range[1] + 1)
    u = np.zeros_like(x)
    for _ in range(n_bumps):
        center = rng.uniform(x[0], x[-1])
        width = rng.uniform(0.02 * L, 0.2 * L)
        a = rng.normal() * amplitude
        u += a * np.exp(-((x - center) ** 2) / (2 * width**2))
    return u


def mixed_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Randomly pick from diverse IC families for training diversity.

    With equal probability, generates one of:
    - Random Fourier (20 modes, power decay 1.5) — smooth multi-scale
    - Single Fourier mode — clean sinusoidal
    - Gaussian bumps (1-4) — localized, unimodal to sparse
    - Low-mode Fourier (1-3 modes) — simple smooth functions

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on [0, L].
    rng : np.random.Generator
        Numpy random generator.
    amplitude : float
        Overall scale factor.

    Returns
    -------
    np.ndarray
        Initial condition, shape ``(len(x),)``.
    """
    family = rng.integers(0, 4)
    if family == 0:
        return random_fourier_1d(x, rng, n_modes=20, power_decay=1.5, amplitude=amplitude)
    elif family == 1:
        # Single Fourier mode
        L = x[-1] - x[0] + (x[1] - x[0])
        k = rng.integers(1, 6)
        a = rng.normal() * amplitude
        phi = rng.uniform(0, 2 * np.pi)
        return a * np.sin(2 * np.pi * k * x / L + phi)
    elif family == 2:
        return gaussian_bumps_1d(x, rng, amplitude=amplitude)
    else:
        # Low-mode Fourier (1-3 modes, no decay)
        n_modes = rng.integers(1, 4)
        return random_fourier_1d(x, rng, n_modes=n_modes, power_decay=0.0, amplitude=amplitude)


def random_fourier_2d(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_modes: int = 10,
    power_decay: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """2D sum of random Fourier modes.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    rng : np.random.Generator
        Numpy random generator.
    n_modes : int
        Number of modes per dimension (total modes = n_modes^2).
    power_decay : float
        Exponent for amplitude decay.
    amplitude : float
        Overall scale.

    Returns
    -------
    np.ndarray
        Initial condition, shape ``(len(x), len(y))``.
    """
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(xx)
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            k_mag = np.sqrt(kx**2 + ky**2)
            a = rng.normal() * amplitude * k_mag ** (-power_decay)
            phi = rng.uniform(0, 2 * np.pi)
            u += a * np.sin(
                2 * np.pi * (kx * xx / Lx + ky * yy / Ly) + phi
            )
    return u


def random_fourier_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
    n_modes: int = 4,
    power_decay: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """3D sum of random Fourier modes (total modes = n_modes^3).

    Same construction as :func:`random_fourier_2d` with a third axis;
    returns shape ``(len(x), len(y), len(z))``.
    """
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0] + (y[1] - y[0])
    Lz = z[-1] - z[0] + (z[1] - z[0])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    u = np.zeros_like(xx)
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            for kz in range(1, n_modes + 1):
                k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
                a = rng.normal() * amplitude * k_mag ** (-power_decay)
                phi = rng.uniform(0, 2 * np.pi)
                u += a * np.sin(
                    2 * np.pi * (kx * xx / Lx + ky * yy / Ly + kz * zz / Lz) + phi
                )
    return u


def tanh_mixture_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    n_transitions: int = 3,
    interface_width: float = 0.05,
) -> np.ndarray:
    """Random piecewise-tanh profile between -1 and +1.

    Suitable for Allen-Cahn initial conditions near the double-well
    metastable states.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on [0, L].
    rng : np.random.Generator
        Numpy random generator.
    n_transitions : int
        Number of transitions between +1 and -1.
    interface_width : float
        Width parameter for each tanh transition.

    Returns
    -------
    np.ndarray
        Initial condition in approximately [-1, 1], shape ``(len(x),)``.
    """
    L = x[-1] - x[0]
    centers = np.sort(rng.uniform(0.1 * L, 0.9 * L, size=n_transitions))
    signs = rng.choice([-1, 1], size=n_transitions)
    u = np.ones_like(x) * rng.choice([-1, 1])
    for c, s in zip(centers, signs):
        u += s * np.tanh((x - c) / interface_width)
    return np.clip(u, -1, 1)


def boundary_driven_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    c_left: float = 1.0,
    decay_length: float = 0.1,
) -> np.ndarray:
    """Smooth initial profile for diffusion-sorption (non-periodic).

    Concentration decays from ``c_left`` at the left boundary.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid on [0, L].
    rng : np.random.Generator
        Numpy random generator (used for slight randomness in decay).
    c_left : float
        Concentration at the left boundary.
    decay_length : float
        Characteristic decay length (fraction of domain).

    Returns
    -------
    np.ndarray
        Non-negative initial condition, shape ``(len(x),)``.
    """
    L = x[-1] - x[0]
    lam = decay_length * L * rng.uniform(0.5, 2.0)
    return c_left * np.exp(-x / lam)


def log_normal_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 4.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Log-normal random permeability field for Darcy flow.

    Generates a Gaussian random field with power spectrum |k|^{-alpha},
    then exponentiates.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    rng : np.random.Generator
        Numpy random generator.
    alpha : float
        Power-law exponent for the spectral decay.
    sigma : float
        Standard deviation in log-space.

    Returns
    -------
    np.ndarray
        Positive permeability field, shape ``(len(x), len(y))``.
    """
    n_x, n_y = len(x), len(y)
    kx = np.fft.fftfreq(n_x, d=(x[1] - x[0]))
    ky = np.fft.fftfreq(n_y, d=(y[1] - y[0]))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    k_mag[0, 0] = 1.0  # avoid division by zero

    # amplitude spectrum
    spectrum = k_mag ** (-alpha / 2)
    spectrum[0, 0] = 0.0  # zero mean in log-space

    # random Fourier coefficients
    noise = rng.standard_normal((n_x, n_y)) + 1j * rng.standard_normal((n_x, n_y))
    field_hat = spectrum * noise
    field = np.real(np.fft.ifft2(field_hat))
    field = sigma * field / (field.std() + 1e-12)  # normalize to target sigma

    return np.exp(field)


def gaussian_bump_2d(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    base_height: float = 1.0,
    bump_height: float = 2.0,
    bump_width: float = 0.1,
) -> np.ndarray:
    """Gaussian bump on a flat background for the shallow-water initial height.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    rng : np.random.Generator
        Numpy random generator.
    base_height : float
        Background water height.
    bump_height : float
        Additional height at the bump center.
    bump_width : float
        Width of the Gaussian bump (fraction of domain).

    Returns
    -------
    np.ndarray
        Water height field, shape ``(len(x), len(y))``.
    """
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    cx = rng.uniform(0.3 * Lx, 0.7 * Lx) + x[0]
    cy = rng.uniform(0.3 * Ly, 0.7 * Ly) + y[0]
    w = bump_width * min(Lx, Ly) * rng.uniform(0.7, 1.3)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return base_height + bump_height * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / w**2)


def gray_scott_ic_2d(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_seeds: int = 5,
    seed_radius: float = 0.05,
) -> np.ndarray:
    """Initial condition for Gray-Scott reaction-diffusion.

    Uniform (u=1, v=0) background with localized seed patches where
    v is elevated and u is depleted.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays.
    rng : np.random.Generator
        Numpy random generator.
    n_seeds : int
        Number of seed patches.
    seed_radius : float
        Radius of each seed (fraction of domain).

    Returns
    -------
    np.ndarray
        Shape ``(2, len(x), len(y))``. Channel 0 = u, channel 1 = v.
    """
    n_x, n_y = len(x), len(y)
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    u = np.ones((n_x, n_y))
    v = np.zeros((n_x, n_y))

    for _ in range(n_seeds):
        cx = rng.uniform(x[0] + 0.2 * Lx, x[0] + 0.8 * Lx)
        cy = rng.uniform(y[0] + 0.2 * Ly, y[0] + 0.8 * Ly)
        r = seed_radius * min(Lx, Ly) * rng.uniform(0.7, 1.3)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
        u[mask] = 0.5 + rng.uniform(-0.05, 0.05)
        v[mask] = 0.25 + rng.uniform(-0.05, 0.05)

    return np.stack([u, v], axis=0)
