"""Hydrogenic bound-free dipoles resolved in orbital angular momentum.

This is an altered finite-n JAX adaptation of the Brocklehurst recurrence in
Cloudy C23.01 hydro_bauman.cpp (Robert Bauman's implementation). Its license
is retained in escape/_licenses/Cloudy.txt. The defining reference is
Brocklehurst (1971), MNRAS 153, 471, equations 3.13 and 3.17--3.24:
https://doi.org/10.1093/mnras/153.4.471 .

Only n<=20 is supported; use float64. Each shell shares one recurrence across
its orbital sublevels to keep the larger-atom compilation tractable.
These are Coulomb-wave dipoles, not fits to the compared level populations.
Collapsed shells average over statistical orbital populations. The same
cross section must be used for photoionization and inverse Milne emission.
"""

from __future__ import annotations

from math import lgamma, log, pi

import jax.numpy as jnp

from .hydrogen_rates import PLANCK_CGS

_BOHR_RADIUS_CM = 5.29177210903e-9
_FINE_STRUCTURE = 1 / 137.035999084
_RESOLVED_NL = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2))
MAX_PRINCIPAL_N = 20


def _orbital_cross_sections(n, relative_energy):
    """Evaluate all l channels using one pair of Coulomb recurrences."""
    energy = jnp.asarray(relative_energy)
    t2 = jnp.maximum(energy - 1, 0.)
    small = t2 < 1e-6
    safe_t = jnp.sqrt(jnp.where(small, 1., t2))
    atan_over_t = jnp.where(small, 1 - t2 / 3 + t2**2 / 5 - t2**3 / 7,
                           jnp.arctan(safe_t) / safe_t)
    k2 = t2 / n**2
    safe_k = jnp.sqrt(jnp.maximum(k2, 1e-30))
    # Remove the common Coulomb normalization and one energy factor per
    # downward step before recursing. Recombine in log space only at the end:
    # squaring the unscaled dipoles underflows for high-l, high-energy tails.
    e = 1 + t2
    log_gk = (.5 * log(pi / 2) + log(8 * n) + n * log(4 * n) - lgamma(2 * n)
              - 2 * n * atan_over_t - .5 * jnp.log(-jnp.expm1(-2 * jnp.pi / safe_k))
              - (n + 2) * jnp.log(e))
    plus = {n - 1: jnp.ones_like(energy)}
    if n > 1:
        plus[n - 2] = (2 * n - 1) * n * plus[n - 1]
    for q in range(n - 3, -1, -1):
        plus[q] = (((4 * n**2 - 4 * (q + 2)**2) / e + (q + 2) * (2 * q + 3))
                   * plus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                   * (1 + (q + 3)**2 * k2) / e**2 * plus[q + 2])
    minus = {n - 1: jnp.ones_like(energy) / (2 * n)}
    if n > 2:
        minus[n - 2] = (2 * n - 1) * (4 / e + n - 1) * minus[n - 1]
    for q in range(n - 3, 0, -1):
        minus[q] = (((4 * n**2 - 4 * (q + 1)**2) / e + (q + 1) * (2 * q + 3))
                    * minus[q + 1] - 4 * n**2 * (n**2 - (q + 2)**2)
                    * (1 + (q + 1)**2 * k2) / e**2 * minus[q + 2])
    log_products = [jnp.zeros_like(energy)]
    for step in range(1, n + 1):
        log_products.append(log_products[-1] + jnp.log1p(step**2 * k2))
    rows = []
    for ell in range(n):
        total = jnp.zeros_like(energy)
        channels = [(ell + 1, plus, n - 1 - ell)]
        if ell > 0:
            channels.append((ell - 1, minus, n - ell))
        for lp, recurrence, power in channels:
            log_radial2 = (lgamma(n + ell + 1) - lgamma(n - ell)
                           + log_products[lp] + 2 * (ell - n) * log(2 * n)
                           + 2 * (log_gk + power * jnp.log(e)
                                  + jnp.log(jnp.abs(recurrence[ell]))))
            total = total + max(ell, lp) / (2 * ell + 1) * e * jnp.exp(log_radial2)
        rows.append(4 / 3 * pi * _FINE_STRUCTURE * _BOHR_RADIUS_CM**2 * n**2 * total)
    sigma = jnp.stack(rows)
    return jnp.where(jnp.isfinite(energy) & (energy > 0),
                     jnp.where(energy >= 1, sigma, 0.), jnp.nan)


def hydrogenic_cross_section(n, ell, relative_energy):
    """Return sigma_nl [cm^2] at photon energy divided by its threshold.

    Quantum numbers are static integers. Below threshold the result is zero;
    the right-hand threshold value is finite. A small-energy series avoids
    a singular intermediate atan(t)/t and retains its derivative.
    """
    if (not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_PRINCIPAL_N
            or not isinstance(ell, int) or isinstance(ell, bool) or not 0 <= ell < n):
        raise ValueError("Use integer quantum numbers 1 <= n <= 20 and 0 <= l < n.")
    return _orbital_cross_sections(n, relative_energy)[ell]


def resolved_atom_cross_sections(atom, frequency_hz):
    """Return (Nlevel,Nfrequency) cross sections for the finite escape atom."""
    frequency = jnp.asarray(frequency_hz)
    count = atom.principal_n.shape[0]
    if not 6 <= count <= MAX_PRINCIPAL_N + 3 or frequency.ndim != 1:
        raise ValueError("Use the six-state atom with collapsed shells through n<=20 and a 1D grid.")
    rows = []
    for i, (n, ell) in enumerate(_RESOLVED_NL):
        rows.append(hydrogenic_cross_section(n, ell, PLANCK_CGS * frequency
                                             / atom.ionization_energy_erg[i]))
    for i in range(6, count):
        n = i - 2
        relative = PLANCK_CGS * frequency / atom.ionization_energy_erg[i]
        weights = (2 * jnp.arange(n) + 1) / n**2
        rows.append(jnp.sum(weights[:, None] * _orbital_cross_sections(n, relative), axis=0))
    return jnp.stack(rows)


__all__ = ["hydrogenic_cross_section", "resolved_atom_cross_sections"]
