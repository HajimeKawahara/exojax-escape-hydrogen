"""Joint radiation/atom/thermal checks, distinct from atomic accuracy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.coupled_hydrogen import CoupledHydrogenModel, atom_line_profiles
from escape.hydrogen_continuum import hydrogen_free_free_coefficients, total_radiative_energy

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def model():
    # Preserve the small-atom numerical benchmark independently of the default.
    return CoupledHydrogenModel(np.linspace(1e10, 1.0001e10, 3), nmax=3, quadrature_order=4)


def test_self_consistent_radiation_populations(model, record_property):
    outer = model.boundary_blackbody(10000., .05)
    inner = jnp.zeros_like(outer)
    density = jnp.full(2, 1e11)
    temperature = jnp.full(2, 9000.)
    solution = model.solve_equilibrium(density, temperature, inner, outer)
    initial = model._initial_fractions(density, temperature, inner, outer)
    assert solution.converged
    np.testing.assert_allclose(solution.bound_cm3.sum(1) + solution.electron_cm3, density, rtol=2e-14)
    assert np.max(np.abs(np.asarray(solution.bound_cm3 / density[:, None] / initial - 1))) > .01
    np.testing.assert_allclose(solution.temperature_k, temperature)
    record_property("joint_equilibrium_residual", float(solution.residual_norm))
    record_property("radiation_balance_error", float(solution.radiation.radiation_balance_error))
    record_property("profile_integral_error", float(jnp.max(jnp.abs(solution.radiation.profile_integrals - 1))))


@pytest.mark.parametrize("nmax", [3, 15])
def test_thin_column_recovers_prescribed_field_population_limit(nmax):
    model = CoupledHydrogenModel(np.array([1e10, 1e10 + 1]), nmax=nmax, quadrature_order=4)
    outer = model.boundary_blackbody(10000., .05)
    inner = jnp.zeros_like(outer)
    density, temperature = jnp.array([1e9]), jnp.array([9000.])
    initial = model._initial_fractions(density, temperature, inner, outer)
    solution = model.solve_equilibrium(density, temperature, inner, outer)
    assert solution.converged
    np.testing.assert_allclose(solution.bound_cm3 / density[:, None], initial, rtol=2e-4)


def test_thermal_equilibrium_solves_temperature(model, record_property):
    outer = model.boundary_blackbody(10000., .05)
    solution = model.solve_equilibrium(1e9, 10000., jnp.zeros_like(outer), outer,
                                       thermal=True, continuation_steps=2)
    assert solution.converged
    assert np.all(np.abs(np.asarray(solution.temperature_k) - 10000.) > 500.)
    fraction = solution.bound_cm3 / 1e9
    profiles = jax.vmap(lambda t: atom_line_profiles(model.atom, t, model.grid)[0])(solution.temperature_k)
    line, cont = model._occupations(solution.radiation.mean_intensity, profiles)
    # Independent rate-based energy calculation, rather than the RT energy sum.
    energy = jax.vmap(lambda t, n, ne, ll, cc: total_radiative_energy(
        model.atom, t, n, ne, ne, ll, cc, model.grid.frequency_hz, model.grid.weight_hz,
    ))(solution.temperature_k, solution.bound_cm3, solution.electron_cm3, line, cont)
    error = float(jnp.max(jnp.abs(energy) / solution.radiation.power_scale_erg_cm3_s))
    assert error < 1e-9
    assert np.all(np.isfinite(fraction))
    record_property("thermal_energy_relative_error", error)
    record_property("thermal_temperature_min_k", float(solution.temperature_k.min()))
    record_property("thermal_temperature_max_k", float(solution.temperature_k.max()))


def test_joint_equilibrium_derivative(model, record_property):
    base = model.boundary_blackbody(10000., .05)
    @jax.jit
    def observable(log_scale):
        density = 1e10 * jnp.array([jnp.exp(log_scale), 1.])
        answer = model.solve_equilibrium(density, 9000., jnp.zeros_like(base), base,
                                         iterations=20, continuation_steps=2)
        # A different cell responds through the solved radiation field.
        return jnp.log(jnp.sum(answer.bound_cm3[1, 1:3]))
    step = 2e-4
    automatic = jax.grad(observable)(0.)
    finite = (observable(step) - observable(-step)) / (2 * step)
    assert abs(float(automatic)) > 1e-6
    error = abs(float((automatic - finite) / finite))
    assert error < 2e-4
    record_property("joint_gradient_relative_error", error)


def test_frequency_refinement(record_property):
    populations = []
    for order in (4, 8):
        model = CoupledHydrogenModel(np.linspace(1e10, 1.0001e10, 3), nmax=3, quadrature_order=order)
        outer = model.boundary_blackbody(10000., .05)
        solution = model.solve_equilibrium(1e10, 9000., jnp.zeros_like(outer), outer,
                                           iterations=20, continuation_steps=2)
        assert solution.converged
        populations.append(np.asarray(solution.bound_cm3[:, 1:].sum(1)))
    change = float(np.max(np.abs(populations[0] / populations[1] - 1)))
    assert change < .01
    record_property("frequency_population_change", change)


def test_frequency_window_resolves_bolometric_free_free(model, record_property):
    _, emission = hydrogen_free_free_coefficients(9000., 1e9, 1e9, model.grid.frequency_hz)
    luminosity = float(4 * jnp.pi * (emission @ model.grid.weight_hz))
    expected = 1.426e-27 * np.sqrt(9000.) * 1e18
    error = abs(luminosity / expected - 1)
    assert error < .001
    record_property("free_free_bolometric_relative_error", error)


def test_invalid_temperature_is_visible(model):
    outer = model.boundary_blackbody(10000., .05)
    solution = model.solve_equilibrium(1e9, 16000., jnp.zeros_like(outer), outer,
                                       iterations=2, continuation_steps=1)
    assert not solution.converged
    assert np.isnan(solution.bound_cm3).all()


@pytest.mark.parametrize("legacy", [False, True], ids=["psm20", "legacy"])
def test_extended_atom_weak_gain_is_explicit(record_property, legacy):
    model = CoupledHydrogenModel(
        np.linspace(1e10, 1.0001e10, 3), nmax=5, quadrature_order=4,
        bound_free_model="kramers" if legacy else "hydrogenic",
        include_orbital_mixing=not legacy,
    )
    outer = model.boundary_blackbody(10000., .05)
    result = model.solve_equilibrium(1e10, 9000., jnp.zeros_like(outer), outer)
    assert result.converged
    atom = model.atom
    weights = (atom.statistical_weights[atom.line_upper] / atom.statistical_weights[atom.line_lower]
               * result.bound_cm3[:, atom.line_lower] - result.bound_cm3[:, atom.line_upper])
    # This point inverts a resolved branch in the explicit legacy atom.
    # The repaired rates remove that inversion; retain both cases so signed
    # transfer coverage does not depend on the new equilibrium having gain.
    assert bool(np.any(np.asarray(weights) < 0)) == legacy
    assert 0 <= result.radiation.gain_optical_depth < .25
    if legacy:
        assert result.radiation.gain_optical_depth > 0
    record_property("extended_atom_gain_depth", float(result.radiation.gain_optical_depth))
    record_property("legacy_atomic_closure", legacy)
