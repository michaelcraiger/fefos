"""Command-line interface.

fefos predict SrTiO3 LiFePO4 CuFeS2 [--mode linear|quadratic|paper]
fefos curves Ti O --q 3
fefos reproduce-paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_MP_VERSION, binary_records
from .model import FEFOS, MODES

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / f"mp_{DEFAULT_MP_VERSION}"


def load_model(data: str | Path = DEFAULT_DATA, mode: str = "linear", window: float = 2.0) -> FEFOS:
    """FEFOS built from the binary compounds of a downloaded MP snapshot."""
    import pandas as pd

    binaries = pd.read_csv(Path(data) / "binaries.csv")
    return FEFOS.from_binary_entries(binary_records(binaries), mode=mode, window=window)


def load_bond_data(data: str | Path = DEFAULT_DATA):
    """Reference bond-length table and fitted correction shipped with a snapshot."""
    import pandas as pd

    from .structure import BondLengthCorrection, BondLengthTable

    data = Path(data)
    table = BondLengthTable.from_frame(pd.read_csv(data / "bond_lengths.csv"))
    correction = BondLengthCorrection.from_frame(pd.read_csv(data / "bond_correction.csv"))
    return table, correction


def _predict_bonds(args) -> None:
    from .structure import predict_with_bonds

    model = load_model(args.data, "linear")
    table, correction = load_bond_data(args.data)
    print(f"{'formula':16s} {'dEf':>8s} {'+bonds':>8s}  assumed M-X bond lengths (charge)")
    for formula in args.formulas:
        r = predict_with_bonds(model, table, correction, formula)
        lengths = ", ".join(
            f"{el} {d:.3f} A ({r.prediction.oxidation_states[el]:+.2f})"
            for el, d in r.features.bond_lengths.items()
        )
        print(
            f"{formula:16s} {r.prediction.formation_energy:8.3f} {r.formation_energy:8.3f}  {lengths}"
        )
    print("energies in eV/atom; +bonds adds the per-anion bond-length correction (linear mode)")


def _predict(args) -> None:
    if args.bonds:
        return _predict_bonds(args)
    model = load_model(args.data, args.mode, args.window)
    print(
        f"{'formula':16s} {'anion':>5s} {'mean OS':>7s} {'dEf':>8s} {'ref':>8s} {'dD':>8s}  oxidation states"
    )
    for formula in args.formulas:
        p = model.predict(formula)
        states = ", ".join(f"{el} {v:+.2f}" for el, v in p.oxidation_states.items())
        print(
            f"{formula:16s} {p.anion:>5s} {p.mean_oxidation_state:7.2f} "
            f"{p.formation_energy:8.3f} {p.reference_energy:8.3f} {p.delta_d:8.3f}  {states}"
        )
    print("energies in eV/atom; dEf = ref + dD")


def _curves(args) -> None:
    model = load_model(args.data, "quadratic" if args.mode == "linear" else args.mode, args.window)
    hull = model.hulls[(args.element, args.anion)]
    print(f"{args.element}–{args.anion} hull vertices (OS, eV/cation):")
    for q, e in zip(hull.vertices_q, hull.vertices_e):
        print(f"  {q:6.3f}  {e:8.3f}")
    r = model.curves(args.element, args.anion, args.q)
    unit = "eV/atom" if r.per_atom else "eV/cation"
    print(f"reference OS {args.q}: E = {r.reference_energy:.3f} {unit}")
    print(f"  f_ox(x)  = {r.ox.a:.4f} x^2 + {r.ox.b:+.4f} x   (0 <= x <= {r.ox.x_max:g})")
    print(f"  f_red(x) = {r.red.a:.4f} x^2 + {r.red.b:+.4f} x   (0 <= x <= {r.red.x_max:g})")


def _reproduce(args) -> None:
    from .legacy import reproduce_published

    rep = reproduce_published()
    diff = (rep.reproduced_prediction - rep.published_prediction).abs()
    err = (rep.reproduced_prediction - rep.mp_formation_energy).abs()
    print(
        f"reproduced {len(rep)} published predictions, max |difference| = {diff.max():.2e} eV/atom"
    )
    print(
        f"MAE vs MP (2022): {err.mean():.3f} eV/atom; coordination-filtered: "
        f"{err[rep.coordination_match].mean():.3f} eV/atom (n={rep.coordination_match.sum()})"
    )


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=DEFAULT_DATA, help="directory containing binaries.csv")
    common.add_argument("--mode", default="linear", choices=MODES)
    common.add_argument("--window", type=float, default=2.0)
    ap = argparse.ArgumentParser(
        prog="fefos", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", parents=[common], help="predict formation energies")
    p.add_argument("formulas", nargs="+")
    p.add_argument(
        "--bonds", action="store_true", help="add assumed bond lengths and the bond correction"
    )
    p.set_defaults(func=_predict)
    c = sub.add_parser(
        "curves", parents=[common], help="show an element's hull and fitted parabolas"
    )
    c.add_argument("element")
    c.add_argument("anion")
    c.add_argument("--q", type=float, required=True, help="reference oxidation state")
    c.set_defaults(func=_curves)
    r = sub.add_parser("reproduce-paper", help="recompute the 2023 published predictions")
    r.set_defaults(func=_reproduce)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
