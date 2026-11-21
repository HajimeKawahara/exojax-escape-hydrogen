"""Independent Coulomb cross-section anchors and stable threshold behavior."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from escape.hydrogen_atom import atom_cross_sections, make_hydrogen_atom
from escape.hydrogen_bound_free import _orbital_cross_sections, hydrogenic_cross_section
from escape.hydrogen_rates import PLANCK_CGS

jax.config.update("jax_enable_x64", True)


def test_coulomb_recurrence_matches_independently_tabulated_anchors(record_property):
    data = json.loads((Path(__file__).parent / "data/hydrogen_bound_free_anchors.json").read_text())
    rows = np.asarray(data["rows"])
    errors = []
    for n in range(1, 6):
        for ell in range(n):
            selected = rows[(rows[:, 0] == n) & (rows[:, 1] == ell)]
            actual = np.asarray(hydrogenic_cross_section(n, ell, np.exp(selected[:, 2])))
            expected = np.exp(selected[:, 3])
            # The archived source rounds logarithms to six decimal places.
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0.)
            errors.extend(abs(actual / expected - 1))
    record_property("hydrogenic_cross_section_anchor_error", float(np.max(errors)))


def test_ground_state_stobbe_expression_and_threshold():
    relative = jnp.geomspace(1.00001, 1e3, 64)
    epsilon = jnp.sqrt(relative - 1)
    threshold = 512 * jnp.pi**2 / 3 / jnp.exp(4) / 137.035999084 * (5.29177210903e-9)**2
    expected = (threshold / relative**4 * jnp.exp(4 - 4 * jnp.arctan(epsilon) / epsilon)
                / (-jnp.expm1(-2 * jnp.pi / epsilon)))
    np.testing.assert_allclose(hydrogenic_cross_section(1, 0, relative), expected, rtol=3e-14)
    assert hydrogenic_cross_section(1, 0, .999) == 0.
    np.testing.assert_allclose(hydrogenic_cross_section(1, 0, 1.), threshold, rtol=3e-14)


def test_high_shells_match_independent_cloudy_cross_section_table(record_property):
    data = json.loads((Path(__file__).parent / "data/hydrogen_high_n_bound_free_anchors.json").read_text())
    rows = np.asarray(data["rows"])
    errors = []
    for n in (6, 8, 10, 15, 20):
        block = rows[rows[:, 0] == n].reshape(n, -1, 6)
        relative = np.exp(block[0, :, 2])
        np.testing.assert_array_equal(block[:, :, 2], np.broadcast_to(block[0, :, 2], block[:, :, 2].shape))
        actual = np.asarray(_orbital_cross_sections(n, relative))
        expected = np.exp(block[:, :, 3])
        # The source rounds both logarithms. Propagate its printed energy
        # precision: high-l tails are steep and amplify that rounding.
        lower = np.log(np.asarray(_orbital_cross_sections(n, np.exp(block[0, :, 2] + block[0, :, 4]))))
        upper = np.log(np.asarray(_orbital_cross_sections(n, np.exp(block[0, :, 2] - block[0, :, 4]))))
        assert np.all(block[:, :, 3] >= lower - block[:, :, 5] - 1e-9)
        assert np.all(block[:, :, 3] <= upper + block[:, :, 5] + 1e-9)
        errors.extend(np.ravel(abs(actual / expected - 1)))
    record_property("high_n_cross_section_anchor_error", float(np.max(errors)))


@pytest.mark.parametrize("n,ell", [(3, 1), (10, 0), (20, 19)])
@pytest.mark.parametrize("relative,step", [(1.000001, 1e-7), (1.01, 1e-5), (10., 1e-3)])
def test_frequency_derivative_matches_finite_difference(n, ell, relative, step):
    function = lambda x: jnp.log(hydrogenic_cross_section(n, ell, x))
    derivative = jax.jit(jax.grad(function))(relative)
    finite = (function(relative + step) - function(relative - step)) / (2 * step)
    np.testing.assert_allclose(derivative, finite, rtol=1e-6, atol=1e-8)


def test_collapsed_shells_are_statistical_orbital_averages():
    atom = make_hydrogen_atom(5)
    relative = jnp.array([1.001, 2., 20.])
    for index, n in ((6, 4), (7, 5)):
        frequency = relative * atom.ionization_energy_erg[index] / PLANCK_CGS
        expected = sum((2 * ell + 1) / n**2 * hydrogenic_cross_section(n, ell, relative)
                       for ell in range(n))
        np.testing.assert_allclose(atom_cross_sections(atom, frequency)[index], expected, rtol=2e-14)


def test_invalid_quantum_numbers_and_model():
    for n, ell in ((0, 0), (21, 0), (3, 3), (True, 0)):
        with pytest.raises(ValueError):
            hydrogenic_cross_section(n, ell, 2.)
    with pytest.raises(ValueError, match="Bound-free model"):
        atom_cross_sections(make_hydrogen_atom(3), jnp.array([1e15]), model="unknown")
