"""Benchmark FEFOS against Materials Project formation energies.

    python scripts/run_benchmark.py [--data data/mp_2026-09-28] [--out results]

Writes ``results/benchmark_predictions.csv.gz`` (one row per compound) and
``results/summary.json`` (error metrics per model and subset), plus a check
that the 2023 published predictions are reproduced.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fefos.benchmark import MODELS, run_benchmark, summarise
from fefos.data import load_dataset
from fefos.legacy import reproduce_published

ALL_MODELS = MODELS + ("fefos_2023",)


def _chunk(args):
    binaries, compounds = args
    return run_benchmark(binaries, compounds)


def chemistry_class(r: pd.DataFrame) -> pd.Series:
    """Ionic (metal cations), polyanion/mixed-anion, or intermetallic-like (mean OS < 1)."""
    return pd.Series(
        np.select(
            [r.mean_oxidation_state < 1, r.nonmetal_cation],
            ["intermetallic-like (mean OS < 1)", "polyanion / mixed anion"],
            "ionic, metal cations",
        ),
        index=r.index,
    )


def subsets(r: pd.DataFrame) -> dict[str, pd.Series]:
    ionic = chemistry_class(r) == "ionic, metal cations"
    return {
        "all representable": r.mean_oxidation_state >= 0,
        "ionic, metal cations": ionic,
        "ionic, metal cations: stable (on MP hull)": ionic & r.is_stable,
        "ionic, metal cations: ICSD (experimental)": ionic & ~r.theoretical,
        "ionic, metal cations: near hull (<0.1 eV/atom)": ionic & (r.energy_above_hull < 0.1),
        "polyanion / mixed anion": chemistry_class(r) == "polyanion / mixed anion",
        "intermetallic-like (mean OS < 1)": chemistry_class(r)
        == "intermetallic-like (mean OS < 1)",
        "paper scope (2-cation oxides, OS 2/3/4)": r.paper_scope,
        "paper scope, 2023 coefficients available": r.paper_scope & r.fefos_2023.notna(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mp_2026-09-28")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    binaries, compounds = load_dataset(args.data)
    shuffled = compounds.sample(frac=1, random_state=0)
    bounds = np.linspace(0, len(shuffled), args.workers * 4 + 1).astype(int)
    chunks = [shuffled.iloc[i:j] for i, j in zip(bounds[:-1], bounds[1:])]
    with Pool(args.workers) as pool:
        parts = pool.map(_chunk, [(binaries, ch) for ch in chunks])
    r = pd.concat(parts).sort_values(["anion", "formula"]).reset_index(drop=True)

    summary = {"n_compounds_total": int(len(compounds)), "n_representable": int(len(r))}
    summary["subsets"] = {
        name: summarise(r[mask], models=ALL_MODELS).to_dict("records")
        for name, mask in subsets(r).items()
    }
    r["chemistry_class"] = chemistry_class(r)
    ionic = r[r.chemistry_class == "ionic, metal cations"]
    summary["by_anion"] = summarise(ionic, ["anion"]).to_dict("records")
    ionic = ionic.assign(n_cat=ionic.n_cations.clip(upper=5))
    summary["by_n_cations"] = summarise(ionic, ["n_cat"]).to_dict("records")
    summary["by_class_and_n_cations"] = summarise(
        r.assign(n_cat=r.n_cations.clip(upper=5)), ["chemistry_class", "n_cat"]
    ).to_dict("records")

    r.to_csv(out / "benchmark_predictions.csv.gz", index=False, compression="gzip")

    rep = reproduce_published()
    diff = (rep.reproduced_prediction - rep.published_prediction).abs()
    err = rep.reproduced_prediction - rep.mp_formation_energy
    summary["paper_2023_reproduction"] = {
        "n": int(len(rep)),
        "max_abs_difference": float(diff.max()),
        "mae_all": float(err.abs().mean()),
        "mae_coordination_filtered": float(err[rep.coordination_match].abs().mean()),
        "n_coordination_filtered": int(rep.coordination_match.sum()),
        "mae_reference_average": float(
            (rep.reference_energy - rep.mp_formation_energy).abs().mean()
        ),
    }
    rep.to_csv(out / "paper_2023_reproduction.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for name, recs in summary["subsets"].items():
        maes = ", ".join(f"{d['model']}={d['mae']:.3f}" for d in recs)
        print(f"{name} (n={recs[0]['n']}): {maes}")


if __name__ == "__main__":
    main()
