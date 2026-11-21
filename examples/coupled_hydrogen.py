"""Compare three closed hydrogen problems on one finite spherical domain.

This is a numerical experiment, not a KELT-9b fit or a transonic solution.
Scientific adequacy and execution gates are recorded separately by MCEG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escape.coupled_hydrogen import CoupledHydrogenModel
from escape.hydrogen_atom import DEFAULT_PRINCIPAL_N, MAX_PRINCIPAL_N


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/coupled_hydrogen"))
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--nmax", type=int, choices=range(3, MAX_PRINCIPAL_N + 1),
                        default=DEFAULT_PRINCIPAL_N)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--width-cm", type=float, default=1e7)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    model = CoupledHydrogenModel(np.linspace(1e10, 1e10 + args.width_cm, args.cells + 1),
                                 nmax=args.nmax, quadrature_order=args.order)
    inner = jnp.zeros_like(model.grid.frequency_hz)
    outer = model.boundary_blackbody(10000., .05)
    density = jnp.full(args.cells, 1e9)
    temperature = jnp.full(args.cells, 9000.)
    prescribed = model.solve_equilibrium(density, temperature, inner, outer)
    thermal = model.solve_equilibrium(density, 10000., inner, outer, thermal=True)
    initial_fractions = model._initial_fractions(density, temperature, inner, outer)[0]
    flow = model.solve_flow(1e9, 9000., 1e6, initial_fractions, inner, outer, 1e21)
    if not bool(prescribed.converged & thermal.converged & flow.converged):
        raise RuntimeError(
            f"Selected example failed: SE={prescribed.residual_norm}, thermal={thermal.residual_norm}, "
            f"flow={flow.residual_norm}; do not treat failed profiles as physical output."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "finite-domain reduced-model experiment; no atmosphere fit or transonic mass-flux selection",
        "nmax": args.nmax, "cells": args.cells, "frequency_points": len(model.grid.frequency_hz),
        "quadrature_order": args.order, "radius_inner_cm": 1e10, "width_cm": args.width_cm,
        "density_base_cm3": 1e9, "temperature_base_k": 9000., "velocity_base_cm_s": 1e6,
        "gravity_gm_cm3_s2": 1e21, "outer_boundary": "I_minus=0.05 B_nu(10000 K)",
        "inner_boundary": "I_plus=0", "hminus_lte_trace": False,
        "se_residual": float(prescribed.residual_norm),
        "thermal_residual": float(thermal.residual_norm),
        "thermal_temperature_k": np.asarray(thermal.temperature_k).tolist(),
        "flow_residual": float(flow.residual_norm),
        "flow_temperature_k": np.asarray(flow.profile.temperature_k).tolist(),
        "flow_material_energy_error": float(flow.material_energy_error),
        "flow_boundary_budget_error": float(flow.energy_balance_error),
        "flow_exchange_relative_error": float(flow.energy_exchange_error),
        "maximum_gain_optical_depth": max(float(case.radiation.gain_optical_depth)
                                           for case in (prescribed, thermal, flow)),
        "profile_integral_error": max(float(jnp.max(jnp.abs(case.radiation.profile_integrals - 1)))
                                       for case in (prescribed, thermal, flow)),
        "not_established": ["atomic truncation accuracy", "thermal stability/uniqueness",
                            "angular closure and moving-line accuracy", "Cloudy agreement"],
    }
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    radius = np.asarray(model.edges_cm)
    distance = (radius[1:] - radius[0]) / 1e5
    flow_bound = np.asarray(flow.profile.n_h_cm3[1:, None] * flow.profile.bound_fraction[1:])
    np.savetxt(args.output / "profiles.csv", np.column_stack((
        distance, prescribed.temperature_k, thermal.temperature_k, flow.profile.temperature_k[1:],
        flow.profile.velocity_cm_s[1:], flow.profile.n_h_cm3[1:], flow.profile.ionized_fraction[1:],
        flow_bound.sum(1), flow_bound[:, 1:3].sum(1), flow_bound[:, 3:6].sum(1),
    )), delimiter=",", comments="", header=(
        "distance_km,T_prescribed_K,T_thermal_K,T_flow_K,velocity_cm_s,nH_cm3,ionized_fraction,"
        "bound_cm3,n2_cm3,n3_cm3"
    ))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    cases = (("Fixed T + solved radiation", prescribed.temperature_k, prescribed.bound_cm3, prescribed.ionized_fraction),
             ("Solved T + radiation", thermal.temperature_k, thermal.bound_cm3, thermal.ionized_fraction),
             ("Advecting flow + radiation", flow.profile.temperature_k[1:], flow_bound, flow.profile.ionized_fraction[1:]))
    for label, temp, bound, ionized in cases:
        bound = np.asarray(bound)
        axes[0, 0].plot(distance, temp, label=label)
        axes[0, 1].plot(distance, 1 - np.asarray(ionized), label=label)
        axes[0, 2].semilogy(distance, bound[:, 1:3].sum(1), label=label)
        axes[1, 2].semilogy(distance, bound[:, 3:6].sum(1), label=label)
    axes[0, 0].set(ylabel="Temperature [K]", title="Three distinct boundary problems")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(ylabel="Neutral fraction", title="Ionization is solved")
    axes[0, 2].set(ylabel="n2 [cm$^{-3}$]", title="Excited population")
    axes[1, 2].set(ylabel="n3 [cm$^{-3}$]", title="H-alpha upper population")
    axes[1, 0].plot(distance, np.asarray(flow.profile.velocity_cm_s[1:]) / 1e5)
    axes[1, 0].set(ylabel="Flow speed [km/s]", title="Momentum + energy + advection")
    axes[1, 1].plot(distance, flow.profile.mean_molecular_weight[1:])
    axes[1, 1].set(ylabel="Mean molecular weight", title="Charge feeds back into pressure")
    axes[1, 1].ticklabel_format(axis="y", useOffset=False)
    for axis in axes.flat:
        axis.set_xlabel("Distance from inner boundary [km]")
        axis.ticklabel_format(axis="x", useOffset=False)
    fig.suptitle(f"Coupled hydrogen experiment: nmax={args.nmax}, {args.cells} cells; not a planetary atmosphere")
    fig.savefig(args.output / "profiles.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
