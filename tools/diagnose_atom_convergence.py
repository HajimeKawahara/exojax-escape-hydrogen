"""Attribute finite-shell population changes using exact block elimination.

The radiation field and low-level atomic coefficients are held fixed. This
provider diagnostic is separate from the runtime model and MCEG's reader.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from escape.hydrogen_atom import atom_rate_coefficients, make_hydrogen_atom
from escape.hydrogen_mixing import mixing_bound_bound_cm3_s
from escape.hydrogen_rates import EV_CGS, PLANCK_CGS, planck_occupation


CONDITIONS = {
    "temperature_k": 9000., "n_h_cm3": 1e12,
    "radiation_temperature_k": 10000., "radiation_dilution": .025,
    "points_per_threshold_interval": 48,
}


def _solve(matrix, source):
    scale = np.diag(matrix)
    return np.linalg.solve(matrix / scale[:, None], source / scale)


def _populations(bound):
    return {"n2": float(bound[1:3].sum()), "n3": float(bound[3:6].sum()),
            "total_bound": float(bound.sum())}


def diagnose():
    """Return fixed-ne Schur budgets and independent pure-H charge solutions."""
    atom = make_hydrogen_atom(20)
    edges = np.unique(np.r_[np.asarray(atom.ionization_energy_erg) / PLANCK_CGS,
                            100 * EV_CGS / PLANCK_CGS])
    x, w = np.polynomial.legendre.leggauss(CONDITIONS["points_per_threshold_interval"])
    width = np.diff(np.log(edges))
    frequency = edges[:-1, None] * np.exp(width[:, None] * (x + 1) / 2)
    weight = frequency * width[:, None] * w / 2
    nu, weight = jnp.asarray(frequency.ravel()), jnp.asarray(weight.ravel())
    temperature = CONDITIONS["temperature_k"]
    radiation_temperature = CONDITIONS["radiation_temperature_k"]
    dilution = CONDITIONS["radiation_dilution"]
    rates = tuple(np.asarray(value) for value in atom_rate_coefficients(
        atom, temperature,
        planck_occupation(atom.line_frequency_hz, radiation_temperature, dilution),
        planck_occupation(nu, radiation_temperature, dilution), nu, weight,
    ))
    radiation, collisions, photo, recombination, ionization, three_body = rates
    low_atom = make_hydrogen_atom(5)
    mixing = jax.jit(lambda ne: mixing_bound_bound_cm3_s(low_atom, temperature, ne))

    def system(nmax, ne):
        count = nmax + 3
        collision = collisions[:count, :count].copy()
        collision[1, 2] = collision[2, 1] = 0.
        low_count = min(count, 8)
        collision[:low_count, :low_count] += np.asarray(mixing(ne))[:low_count, :low_count]
        radiative = radiation[:count, :count]
        transition = radiative + ne * collision
        loss = transition.sum(1) + photo[:count] + ne * ionization[:count]
        source = ne * recombination[:count] + ne**2 * three_body[:count]
        matrix = np.diag(loss) - transition.T
        return matrix, source, transition, loss, radiative, ne * collision

    def fixed(nmax, ne):
        matrix, source, *_ = system(nmax, ne)
        return ne * _solve(matrix, source)

    def closed(nmax):
        def residual(z):
            ne = CONDITIONS["n_h_cm3"] * np.exp(z)
            matrix, source, *_ = system(nmax, ne)
            return z + np.log1p(_solve(matrix, source).sum())

        z = brentq(residual, -40., 0., xtol=5e-15)
        ne = CONDITIONS["n_h_cm3"] * np.exp(z)
        return ne, fixed(nmax, ne), residual(z)

    models = {}
    for nmax in (4, 5, 10, 15, 20):
        ne, bound, residual = closed(nmax)
        matrix, source, transition, loss, radiative, collisional = system(nmax, ne)
        net = ne * source - matrix @ bound
        scale = ne * source + transition.T @ bound + loss * bound

        def transfer(indices):
            target, indices = np.arange(3, 6), np.asarray(indices, dtype=int)

            def flux(block):
                return {
                    "in": float(np.sum(bound[indices, None] * block[np.ix_(indices, target)])),
                    "out": float(np.sum(bound[target, None] * block[np.ix_(target, indices)])),
                }

            rad, col = flux(radiative), flux(collisional)
            return {"radiative": rad, "collisional": col,
                    "net": rad["in"] + col["in"] - rad["out"] - col["out"]}

        def continuum(indices):
            indices = np.asarray(indices, dtype=int)
            terms = {
                "radiative_recombination": float(ne**2 * recombination[indices].sum()),
                "three_body_recombination": float(ne**3 * three_body[indices].sum()),
                "photoionization": float((bound[indices] * photo[indices]).sum()),
                "collisional_ionization": float(ne * (bound[indices] * ionization[indices]).sum()),
            }
            terms["net"] = (terms["radiative_recombination"] + terms["three_body_recombination"]
                            - terms["photoionization"] - terms["collisional_ionization"])
            return terms

        high = np.arange(8, nmax + 3)
        conservation = abs((ne + bound.sum()) / CONDITIONS["n_h_cm3"] - 1)
        scaled_residual = np.max(np.abs(net) / scale)
        assert abs(residual) < 1e-10 and conservation < 1e-10 and scaled_residual < 1e-10
        models[str(nmax)] = {
            "electron_cm3": float(ne), **_populations(bound),
            "charge_residual": float(residual), "conservation_error": float(conservation),
            "max_relative_rate_residual": float(scaled_residual),
            "n3_flux_budget_cm3_s": {
                "n1": transfer([0]), "n2": transfer([1, 2]),
                "n4_5": transfer(range(6, min(nmax + 3, 8))), "n6plus": transfer(high),
                "continuum": continuum(np.arange(3, 6)),
            },
            "n6plus_continuum_budget_cm3_s": continuum(high),
        }

    comparisons = {}
    for nlow, nmax in ((4, 5), (5, 10), (5, 15), (5, 20)):
        cut = nlow + 3
        ne = models[str(nlow)]["electron_cm3"]
        matrix_small, source_small, *_ = system(nlow, ne)
        ratios_small = _solve(matrix_small, source_small)
        bound_small = ne * ratios_small
        matrix, source, *_ = system(nmax, ne)
        ll, lh = matrix[:cut, :cut], matrix[:cut, cut:]
        hl, hh = matrix[cut:, :cut], matrix[cut:, cut:]
        effective = ll - lh @ np.linalg.solve(hh, hl)
        added_source = -lh @ np.linalg.solve(hh, source[cut:])
        reduced = ne * _solve(effective, source[:cut] + added_source)
        full = fixed(nmax, ne)
        source_effect = ne * _solve(effective, added_source)
        operator_effect = -ne * _solve(effective, (effective - matrix_small) @ ratios_small)
        np.testing.assert_allclose(reduced, full[:cut], rtol=2e-12)
        np.testing.assert_allclose(reduced - bound_small, source_effect + operator_effect,
                                   rtol=2e-11, atol=1e-5)
        radiative_source = -lh @ np.linalg.solve(hh, ne * recombination[cut:nmax + 3])
        three_body_source = -lh @ np.linalg.solve(hh, ne**2 * three_body[cut:nmax + 3])
        entry = {}
        for label, indices in (("n2", slice(1, 3)), ("n3", slice(3, 6))):
            baseline, fixed_value = bound_small[indices].sum(), reduced[indices].sum()
            actual = models[str(nmax)][label]
            entry[label] = {
                "baseline_cm3": float(baseline), "full_at_baseline_ne_cm3": float(fixed_value),
                "full_with_charge_closure_cm3": actual,
                "delta_from_added_recombination_cm3": float(source_effect[indices].sum()),
                "delta_from_added_radiative_recombination_cm3":
                    float(ne * _solve(effective, radiative_source)[indices].sum()),
                "delta_from_added_three_body_recombination_cm3":
                    float(ne * _solve(effective, three_body_source)[indices].sum()),
                "delta_from_effective_operator_cm3": float(operator_effect[indices].sum()),
                "delta_from_charge_feedback_cm3": float(actual - fixed_value),
                "relative_fixed_ne_increase": float(fixed_value / baseline - 1),
                "relative_total_increase": float(actual / baseline - 1),
            }
        entry["schur_full_max_relative_difference"] = float(np.max(np.abs(reduced / full[:cut] - 1)))
        entry["low_source_max_relative_difference"] = float(np.max(np.abs(source[:cut] / source_small - 1)))
        comparisons[f"{nlow}_to_{nmax}"] = entry
    return {"models": models, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; refuse to overwrite.")
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("Use a new output path to preserve earlier diagnoses.")
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    inputs = [root / "src/escape" / name for name in (
        "hydrogen_atom.py", "hydrogen_bound_free.py", "hydrogen_mixing.py",
        "hydrogen_nlte.py", "hydrogen_rates.py", "parker.py",
    )] + [Path(__file__).resolve()]
    hashes = lambda: {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in inputs}
    before, start = hashes(), time.monotonic()
    result = diagnose()
    after = hashes()
    if before != after:
        raise RuntimeError("Scientific source changed during the diagnostic execution.")
    record = {
        "kind": "fixed_electron_density_schur_diagnosis", "conditions": CONDITIONS,
        "argv": sys.argv, "python": platform.python_version(), "jax": jax.__version__,
        "elapsed_seconds": time.monotonic() - start, "selected_files_before": before,
        "selected_files_after": after, "selected_files_unchanged_during_run": True,
        "method": [
            "Evaluate nmax20 coefficients on a shared threshold-split grid; exact subsets define smaller atoms without changing low-level rates.",
            "Solve charge conservation with scipy.brentq and scaled NumPy linear solves; evaluate PSM20 mixing at every trial density.",
            "At the smaller-atom ne, eliminate added levels: M_eff=M_LL-M_LH inv(M_HH) M_HL; s_eff=s_L-M_LH inv(M_HH) s_H.",
            "The exact fixed-ne difference is ne inv(M_eff)[delta_s-delta_M b_small]. Split delta_s into radiative and three-body sources.",
            "Charge feedback is the full-atom change when ne is allowed to satisfy its own conservation equation.",
        ],
        **result,
        "limitations": [
            "Source/operator attribution is a defined diagnostic about the smaller atom at fixed ne; the terms are not separate self-consistent atmospheres.",
            "Large three-body and collisional-ionization contributions nearly cancel. Assess net continuum and bound-bound flows before interpreting gross source terms.",
            "No fitted correction, topoff, orbital-collapse validation or infinite-atom convergence claim is introduced.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({n: {key: row[key] for key in ("electron_cm3", "n2", "n3")}
                      for n, row in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
