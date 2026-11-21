"""Data anchors, finite-atom thermodynamic identities and cascade closure."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gammaln

from escape.hydrogen_atom import (
    HYDROGEN_BOHR_RADIUS_CM, MAX_PRINCIPAL_N, atom_bound_free_rates, atom_collision_rates,
    atom_rate_coefficients, atom_saha_factors, hydrogenic_einstein_a,
    hydrogenic_radial_dipole, make_hydrogen_atom, solve_atom_populations,
)
from escape.hydrogen_nlte import solve_statistical_equilibrium
from escape.hydrogen_rates import (
    BOLTZMANN_CGS, EV_CGS, LINE_EINSTEIN_A_S, PLANCK_CGS,
    blackbody_hydrogen_rates, collision_rates, planck_occupation,
)

jax.config.update("jax_enable_x64", True)


def continuum_grid(atom, points_per_interval=48):
    """Independent quadrature split at every threshold to avoid edge error."""
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            100 * EV_CGS / PLANCK_CGS])
    x, w = np.polynomial.legendre.leggauss(points_per_interval)
    widths = np.diff(np.log(edges))
    frequency = edges[:-1, None] * np.exp(widths[:, None] * (x + 1) / 2)
    weights = frequency * widths[:, None] * w / 2
    return jnp.asarray(frequency.ravel()), jnp.asarray(weights.ravel())


def test_radial_integral_and_published_transition_anchors():
    # Analytic 1s--2p radial integral; published A values are rounded to 3 digits.
    assert hydrogenic_radial_dipole(2, 1, 1, 0) / HYDROGEN_BOHR_RADIUS_CM == pytest.approx(
        256 / (81 * np.sqrt(6)), rel=2e-14,
    )
    assert hydrogenic_einstein_a(2, 1, 1, 0) == pytest.approx(6.2648e8, rel=1e-4)
    # Independent published n <= 5 radiative anchors, Table A1:
    # https://academic.oup.com/view-large/375560502
    assert hydrogenic_einstein_a(4, 1, 3, 0) == pytest.approx(3.07e6, rel=.003)
    assert hydrogenic_einstein_a(5, 3, 3, 2) == pytest.approx(4.54e6, rel=.003)
    assert hydrogenic_einstein_a(5, 4, 4, 3) == pytest.approx(4.26e6, rel=.003)
    assert hydrogenic_einstein_a(4, 2, 2, 0) == 0


@pytest.mark.parametrize("nmax,levels,lines", [
    (3, 6, 5), (4, 7, 11), (5, 8, 18), (10, 13, 68), (20, 23, 243),
])
def test_truncation_keeps_low_data_and_adds_explicit_cascade_edges(nmax, levels, lines):
    atom = make_hydrogen_atom(nmax)
    assert atom.statistical_weights.shape == (levels,)
    assert atom.einstein_a_s.shape == (lines,)
    np.testing.assert_array_equal(atom.einstein_a_s[:5], LINE_EINSTEIN_A_S)
    if nmax >= 4:
        assert atom.statistical_weights[6] == 32
        assert np.any((atom.line_upper == 6) & (atom.line_lower == 3))
        # The collapsed H-beta shell A is independently tabulated near 8.419e6.
        beta = (atom.line_upper == 6) & (atom.principal_n[atom.line_lower] == 2)
        assert float(jnp.sum(atom.einstein_a_s[beta])) == pytest.approx(8.419e6, rel=2e-4)
    if nmax >= 5:
        assert atom.statistical_weights[7] == 50
        assert np.any((atom.line_upper == 7) & (atom.line_lower == 6))


@pytest.mark.parametrize("upper,lower", [
    ((6, 1), (1, 0)), ((10, 8), (9, 7)),
    ((20, 0), (19, 1)), ((20, 10), (12, 9)),
])
def test_high_shell_dipoles_match_independent_radial_quadrature(upper, lower):
    """Integrate normalized orbital functions, without the rational expansion."""
    inverse_scale = 1 / upper[0] + 1 / lower[0]

    def orbital_polynomial(n, ell, radius):
        normalization = 2 / n**2 * np.exp(
            (gammaln(n - ell) - gammaln(n + ell + 1)) / 2,
        )
        return (normalization * (2 * radius / n)**ell
                * eval_genlaguerre(n - ell - 1, 2 * ell + 1, 2 * radius / n))

    def integrand(coordinate):
        radius = coordinate / inverse_scale
        return (orbital_polynomial(*upper, radius) * orbital_polynomial(*lower, radius)
                * np.exp(-coordinate) * radius**3 / inverse_scale)

    expected, error = quad(integrand, 0., 250., epsabs=1e-10, epsrel=2e-11, limit=150)
    assert error < abs(expected) * 2e-10
    actual = hydrogenic_radial_dipole(*upper, *lower) / HYDROGEN_BOHR_RADIUS_CM
    assert actual == pytest.approx(expected, rel=2e-10)


@pytest.mark.parametrize("nmax", [5, 10, 20])
@pytest.mark.parametrize("temperature", [6000., 10000., 14000.])
def test_extended_collisions_preserve_all_pairwise_equilibria(temperature, nmax):
    atom = make_hydrogen_atom(nmax)
    bb, ion, three = atom_collision_rates(atom, temperature)
    old_bb, old_ion, _ = collision_rates(temperature)
    np.testing.assert_array_equal(bb[:6, :6], old_bb)
    np.testing.assert_array_equal(ion[:6], old_ion)
    assert np.isfinite(bb).all() and np.asarray(bb >= 0).all()
    boltzmann = np.asarray(atom.statistical_weights) * np.exp(
        -np.asarray(atom.excitation_energy_erg) / (BOLTZMANN_CGS * temperature)
    )
    flux = boltzmann[:, None] * bb
    np.testing.assert_allclose(flux, flux.T, rtol=2e-13, atol=1e-30)
    np.testing.assert_allclose(three / ion, atom_saha_factors(atom, temperature), rtol=2e-13)


def test_extra_shells_preserve_previous_low_level_rate_data():
    smaller, larger = make_hydrogen_atom(5), make_hydrogen_atom(20)
    for temperature in (6000., 9000., 14000.):
        old_bb, old_ion, old_three = atom_collision_rates(smaller, temperature)
        new_bb, new_ion, new_three = atom_collision_rates(larger, temperature)
        np.testing.assert_array_equal(new_bb[:8, :8], old_bb)
        np.testing.assert_array_equal(new_ion[:8], old_ion)
        np.testing.assert_array_equal(new_three[:8], old_three)


@pytest.mark.parametrize("n,temperature,expected", [
    (6, 6000., 3.0976659751338506e-6),
    (10, 9000., 2.5098090298669288e-5),
    (20, 14000., 1.495874742673897e-4),
])
def test_high_shell_ionization_matches_cloudy_vs80_source(n, temperature, expected):
    """Anchors evaluated by compiling C23.01 hydro_vs_ioniz unchanged.

    Input binding energy is 13.5984345/n**2 eV, with this model's exact kB/eV
    conversion. No highest-level topoff multiplier is applied. Formula: VS80
    Eq. 8; source: https://data.nublado.org/cloudy_releases/c23/c23.01.tar.gz
    (source/hydro_vs_rates.cpp, with its license retained in this package).
    """
    ionization = atom_collision_rates(make_hydrogen_atom(20), temperature)[1]
    assert float(ionization[n + 2]) == pytest.approx(expected, rel=2e-14)


def test_shared_grid_reproduces_old_atom_rates_without_edge_bias():
    atom = make_hydrogen_atom(3)
    nu, weights = continuum_grid(atom)
    rates = atom_rate_coefficients(
        atom, 8000., planck_occupation(atom.line_frequency_hz, 10000., .025),
        planck_occupation(nu, 10000., .025), nu, weights, bound_free_model="kramers",
    )
    reference = blackbody_hydrogen_rates(8000., 10000., .025)
    for actual, expected in zip(rates, reference):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-38)


def test_extended_milne_identity_and_differentiable_population_solve():
    atom = make_hydrogen_atom(5)
    nu, weights = continuum_grid(atom, 32)
    gamma, alpha = atom_bound_free_rates(atom, 9000., planck_occupation(nu, 9000.), nu, weights)
    np.testing.assert_allclose(alpha / gamma, atom_saha_factors(atom, 9000.), rtol=2e-13)

    def observable(parameters):
        temperature, dilution = 9000 * parameters[0], .025 * parameters[1]
        rates = atom_rate_coefficients(
            atom, temperature, planck_occupation(atom.line_frequency_hz, 10000., dilution),
            planck_occupation(nu, 10000., dilution), nu, weights,
        )
        result = solve_statistical_equilibrium(1e12, rates)
        return jnp.log(jnp.sum(result.bound_cm3[1:3]))

    point, step = jnp.ones(2), 1e-4
    automatic = jax.jit(jax.grad(observable))(point)
    finite = jnp.array([(observable(point + step * d) - observable(point - step * d)) / (2 * step)
                        for d in jnp.eye(2)])
    assert np.isfinite(automatic).all()
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-9)


def test_atom_truncation_sensitivity(record_property):
    """Record truncation counterevidence separately from numerical correctness.

    MCEG applies the physical adequacy threshold to the saved metrics. This
    test only establishes that the comparison was solved and measured; a
    passing pytest result must not be read as convergence in atom size.
    """
    conditions = {"temperature_k": 9000., "n_h_cm3": 1e12,
                  "radiation_temperature_k": 10000., "radiation_dilution": .025,
                  "continuum_points_per_interval": 48}
    for name, value in conditions.items():
        record_property(name, value)
    populations = []
    for nmax in (3, 4, 5):
        atom = make_hydrogen_atom(nmax)
        nu, weights = continuum_grid(atom, conditions["continuum_points_per_interval"])
        result = solve_atom_populations(
            conditions["n_h_cm3"], atom, conditions["temperature_k"],
            planck_occupation(atom.line_frequency_hz, conditions["radiation_temperature_k"],
                              conditions["radiation_dilution"]),
            planck_occupation(nu, conditions["radiation_temperature_k"],
                              conditions["radiation_dilution"]), nu, weights,
        )
        assert result.converged
        n2 = float(jnp.sum(result.bound_cm3[1:3]))
        n3 = float(jnp.sum(result.bound_cm3[3:6]))
        assert np.isfinite([n2, n3]).all() and n2 > 0 and n3 > 0
        record_property(f"n2_cm3_nmax_{nmax}", n2)
        record_property(f"n3_cm3_nmax_{nmax}", n3)
        record_property(f"ionized_fraction_nmax_{nmax}", float(result.ionized_fraction))
        populations.append((n2, n3))
    record_property("n2_relative_change_4_to_5", abs(populations[2][0] / populations[1][0] - 1))
    record_property("n3_relative_change_4_to_5", abs(populations[2][1] / populations[1][1] - 1))


def test_invalid_atomic_choices_and_temperature_are_not_extrapolated():
    with pytest.raises(ValueError, match="nmax"):
        make_hydrogen_atom(MAX_PRINCIPAL_N + 1)
    with pytest.raises(ValueError, match="nmax"):
        make_hydrogen_atom(True)
    for temperature in (5000., 14001.):
        assert np.isnan(atom_collision_rates(make_hydrogen_atom(5), temperature)[0]).all()
