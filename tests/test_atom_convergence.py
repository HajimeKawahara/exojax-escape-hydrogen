"""Measure shell truncation separately from quadrature and solver derivatives."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from escape.hydrogen_atom import make_hydrogen_atom, solve_atom_populations
from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation

jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=None)
def _inputs(nmax, order):
    atom = make_hydrogen_atom(nmax)
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            100 * EV_CGS / PLANCK_CGS])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    widths = np.diff(np.log(edges))
    frequency = edges[:-1, None] * np.exp(widths[:, None] * (nodes + 1) / 2)
    integration_weights = frequency * widths[:, None] * weights / 2
    frequency = jnp.asarray(frequency.ravel())
    return (atom, frequency, jnp.asarray(integration_weights.ravel()),
            planck_occupation(atom.line_frequency_hz, 10000., .025),
            planck_occupation(frequency, 10000., .025))


def _solve(nmax, order=48, *, temperature=9000., density=1e12):
    atom, frequency, weights, line_occupation, continuum_occupation = _inputs(nmax, order)
    return solve_atom_populations(density, atom, temperature, line_occupation,
                                  continuum_occupation, frequency, weights)


def _low_populations(result):
    return jnp.array([jnp.sum(result.bound_cm3[1:3]),
                      jnp.sum(result.bound_cm3[3:6])])


def test_enlarged_atom_shell_sensitivity(record_property):
    """Measure valid solutions; the provider gate applies the five-percent target."""
    for name, value in {"temperature_k": 9000., "n_h_cm3": 1e12,
                        "radiation_temperature_k": 10000., "radiation_dilution": .025,
                        "continuum_points_per_interval": 48}.items():
        record_property(name, value)
    populations = {}
    for nmax in (5, 10, 15, 20):
        result = _solve(nmax)
        assert result.converged
        values = np.asarray(_low_populations(result))
        assert np.isfinite(values).all() and (values > 0).all()
        populations[nmax] = values
        for shell, value in zip((2, 3), values):
            record_property(f"n{shell}_cm3_nmax_{nmax}", float(value))
        record_property(f"ionized_fraction_nmax_{nmax}", float(result.ionized_fraction))
        record_property(f"rate_residual_norm_nmax_{nmax}", float(result.rate_residual_norm))
        record_property(f"conservation_error_nmax_{nmax}", float(result.conservation_error))
    for lower, upper in ((5, 10), (10, 15), (15, 20), (10, 20), (5, 20)):
        changes = np.abs(populations[upper] / populations[lower] - 1)
        for shell, change in zip((2, 3), changes):
            record_property(f"n{shell}_relative_change_{lower}_to_{upper}", float(change))


def test_enlarged_atom_quadrature_refinement(record_property):
    """Continuum integration error must be small compared with shell sensitivity."""
    ordinary, refined = _solve(20, 48), _solve(20, 96)
    assert ordinary.converged and refined.converged
    changes = np.abs(np.asarray(_low_populations(refined) / _low_populations(ordinary) - 1))
    for shell, change in zip((2, 3), changes):
        record_property(f"n{shell}_quadrature_change_nmax20", float(change))
    assert np.isfinite(changes).all()
    assert np.max(changes) < 1e-3


def test_enlarged_atom_implicit_derivatives(record_property):
    """Differentiate low populations through the full density-dependent charge root."""
    atom, frequency, weights, line_occupation, continuum_occupation = _inputs(20, 48)

    def observable(parameters):
        result = solve_atom_populations(
            1e12 * jnp.exp(parameters[1]), atom, 9000 * jnp.exp(parameters[0]),
            line_occupation, continuum_occupation, frequency, weights,
        )
        return jnp.log(_low_populations(result))

    point, step = jnp.zeros(2), 1e-4
    automatic = np.asarray(jax.jit(jax.jacfwd(observable))(point))
    reverse = np.asarray(jax.jit(jax.jacrev(observable))(point))
    finite = np.column_stack([
        np.asarray((observable(point + step * direction)
                    - observable(point - step * direction)) / (2 * step))
        for direction in jnp.eye(2)
    ])
    assert all(np.isfinite(values).all() for values in (automatic, reverse, finite))
    normalized_error = max(
        np.max(np.abs(values - finite) / np.maximum(1., np.abs(finite)))
        for values in (automatic, reverse)
    )
    record_property("population_gradient_normalized_error_nmax20", float(normalized_error))
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(reverse, finite, rtol=2e-5, atol=1e-8)
