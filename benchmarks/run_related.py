"""Benchmark FEFOS against composition-based machine learning on the same data.

    python benchmarks/features.py
    python benchmarks/run_related.py [--roost]   # -> results/related_work.json

Protocol
* Scope: ionic compounds with metal cations (mean OS >= 1, no nonmetal cation).
* Test set: every compound in 20% of chemical systems (cation sets), so no
  test chemistry is ever seen in training, at any composition.
* Training: all binary compositions (FEFOS uses them, so ML gets them too) plus
  N multi-cation compounds from the other chemical systems, N = 0 ... all.
  N = 0 gives ML exactly the information FEFOS has.
* Metrics: formation-energy MAE, and the stability metrics of Bartel et al.
  (npj Comput. Mater. 6, 97, 2020): the predicted decomposition energy
  dHd = dEf(pred) - E_hull(all other MP phases) classifies each compound as
  stable (dHd < -1 meV/atom) or not; reported as precision, recall, F1 and
  ROC-AUC against the same quantity from MP energies. The hull of the other
  phases is recomputed from MP formation energies (fefos.phase).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fefos import IonicComposition
from fefos.cli import load_model
from fefos.data import DEFAULT_MP_VERSION, load_dataset, parse_composition_column
from fefos.structure import BondLengthCorrection, BondLengthTable, bond_features

sys.path.insert(0, str(Path(__file__).parent))
from features import composition_key  # noqa: E402

from fefos.phase import HullCalculator, mp_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / f"mp_{DEFAULT_MP_VERSION}"
CACHE = ROOT / "benchmarks" / ".cache"
SIZES = [0, 100, 300, 1000, 3000, 10000, 30000, None]  # None = whole training pool
STABLE_TOL = -0.001


def parse_states(s):
    return {k: float(v) for k, v in (p.split(":") for p in s.split(";"))}


def build_tables():
    binaries, compounds = load_dataset(DATA)
    res = pd.read_csv(ROOT / "results" / "benchmark_predictions.csv.gz")
    res = res[(res.mean_oxidation_state >= 1) & ~res.nonmetal_cation].copy()
    comp = compounds.set_index("material_id")
    res["composition"] = comp.composition.reindex(res.material_id).to_numpy()
    table = BondLengthTable.from_reference_bonds(pd.read_csv(DATA / "binary_bonds.csv"))
    rows = []
    for r in res.itertuples(index=False):
        c = parse_composition_column(r.composition)
        ion = IonicComposition.from_composition(c, r.anion)
        if not all(table.has(el, r.anion) for el in ion.cations):
            continue
        f = bond_features(table, r.anion, ion.cations, parse_states(r.oxidation_states))
        rows.append(
            {
                "material_id": r.material_id,
                "formula": r.formula,
                "anion": r.anion,
                "n_cations": r.n_cations,
                "chemsys": "-".join(sorted(ion.cations)),
                "key": composition_key(c),
                "comp": c,
                "mp": r.mp,
                "fefos": r.fefos_linear,
                "reference": r.reference_average,
                "mean_os": r.mean_oxidation_state,
                "is_stable": r.is_stable,
                "theoretical": r.theoretical,
                "e_hull": r.energy_above_hull,
                "phi": f.ionic_potential_spread,
                "size": f.size_mismatch,
            }
        )
    multi = pd.DataFrame(rows)
    multi["fold"] = [zlib.crc32(s.encode()) % 5 for s in multi.chemsys]
    # binaries: lowest polymorph per composition, FEFOS = hull energy at that composition
    model = load_model(DATA)
    b = binaries.assign(r=(binaries.n_anion / binaries.n_element).round(6))
    b = b.sort_values("formation_energy_per_atom").drop_duplicates(["element", "anion", "r"])
    brow = []
    for r in b.itertuples(index=False):
        c = {r.element: r.n_element, r.anion: r.n_anion}
        if not model.can_represent(c, r.anion):
            continue
        p = model.predict(c, r.anion)
        brow.append(
            {
                "key": composition_key(c),
                "comp": c,
                "anion": r.anion,
                "mp": r.formation_energy_per_atom,
                "fefos": p.formation_energy,
                "reference": p.reference_energy,
                "mean_os": p.mean_oxidation_state,
                "n_cations": 1,
                "phi": 0.0,
                "size": 0.0,
            }
        )
    return multi, pd.DataFrame(brow)


def scores(pred, test: pd.DataFrame) -> dict:
    mp = test.mp.to_numpy()
    err = np.abs(pred - mp)
    hull_others = mp - test.dhd.to_numpy()
    dhd_pred = pred - hull_others
    y_true = test.is_stable.to_numpy()
    y_pred = dhd_pred < STABLE_TOL
    out = {
        "mae": float(err.mean()),
        "mae_stable": float(err[y_true].mean()),
        "mae_icsd": float(err[~test.theoretical.to_numpy()].mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, -dhd_pred)),
        "predicted_stable_fraction": float(y_pred.mean()),
    }
    return {k: round(v, 4) for k, v in out.items()}


def fit_correction(train: pd.DataFrame) -> BondLengthCorrection:
    """Per-anion bond correction; anions with few samples share a global fit."""
    near = train[train.e_hull < 0.1]
    X = near[["phi", "size"]].to_numpy()
    y = (near.mp - near.fefos).to_numpy()
    per = BondLengthCorrection.fit(near.anion, X, y, min_samples=30)
    if len(near) >= 10:
        glob = BondLengthCorrection.fit(np.array(["*"] * len(near)), X, y, min_samples=10)
        for an in train.anion.unique():
            per.coefficients.setdefault(an, glob.coefficients["*"])
    return per


def apply_correction(corr, df):
    c = np.array([corr.coefficients.get(a, np.zeros(3)) for a in df.anion])
    return df.fefos.to_numpy() + c[:, 0] + np.sum(c[:, 1:] * df[["phi", "size"]].to_numpy(), axis=1)


def hgb(X, y, seed):
    m = HistGradientBoostingRegressor(
        max_iter=2000,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=seed,
    )
    return m.fit(X, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roost", action="store_true", help="also train the Roost-style network")
    ap.add_argument("--roost-epochs", type=int, default=60)
    args = ap.parse_args()

    multi, bins = build_tables()
    magpie = pd.read_parquet(CACHE / "magpie.parquet")
    test = multi[multi.fold == 0].reset_index(drop=True)
    hulls = HullCalculator.from_frame(mp_entries(DEFAULT_MP_VERSION, CACHE))
    test["hull_others"] = [hulls.hull_energy(c, exclude_self=True) for c in test.comp]
    test["dhd"] = test.mp - test.hull_others
    test["is_stable"] = test.dhd < 0
    test[["material_id", "formula", "mp", "hull_others", "dhd"]].to_csv(
        ROOT / "results" / "related_work_test_hulls.csv.gz", index=False, compression="gzip"
    )
    pool = multi[multi.fold != 0].reset_index(drop=True)
    extra = ["fefos", "reference", "mean_os", "phi", "size", "n_cations"]

    def X_magpie(df):
        return magpie.loc[df.key].to_numpy()

    def X_delta(df):
        return np.hstack([X_magpie(df), df[extra].to_numpy(float)])

    results = {
        "n_test": int(len(test)),
        "n_pool": int(len(pool)),
        "n_binaries": int(len(bins)),
        "n_test_stable": int(test.is_stable.sum()),
        "curves": {},
    }
    curves = results["curves"]
    curves["FEFOS"] = {"0": scores(test.fefos.to_numpy(), test)}
    curves["reference average"] = {"0": scores(test.reference.to_numpy(), test)}
    for name in (
        "FEFOS + bond correction",
        "Magpie + gradient boosting",
        "FEFOS + Magpie gradient boosting (residual)",
    ):
        curves[name] = {}
    if args.roost:
        import roost_lite

        curves["Roost-style network"] = {}

    for size in SIZES:
        n = len(pool) if size is None else size
        seeds = [0, 1, 2] if n <= 3000 else [0]
        acc: dict[str, list] = {k: [] for k in curves if k not in ("FEFOS", "reference average")}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sub = pool.iloc[rng.choice(len(pool), n, replace=False)] if n else pool.iloc[[]]
            train = pd.concat([bins.assign(e_hull=0.0), sub], ignore_index=True)
            t0 = time.time()
            if n >= 30:
                acc["FEFOS + bond correction"].append(
                    scores(apply_correction(fit_correction(sub), test), test)
                )
            m = hgb(X_magpie(train), train.mp.to_numpy(), seed)
            acc["Magpie + gradient boosting"].append(scores(m.predict(X_magpie(test)), test))
            m = hgb(X_delta(train), (train.mp - train.fefos).to_numpy(), seed)
            acc["FEFOS + Magpie gradient boosting (residual)"].append(
                scores(test.fefos.to_numpy() + m.predict(X_delta(test)), test)
            )
            if args.roost and seed == 0 and size in (0, 1000, 10000, None):
                p = roost_lite.train_predict(
                    list(train.comp),
                    train.mp.to_numpy(),
                    list(test.comp),
                    epochs=args.roost_epochs,
                    seed=seed,
                )
                acc["Roost-style network"].append(scores(p, test))
            print(f"N={n} seed={seed} {time.time() - t0:.0f}s", flush=True)
        for name, runs in acc.items():
            if runs:
                curves[name][str(n)] = {
                    k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]
                }
        print(json.dumps({k: v.get(str(n), {}).get("mae") for k, v in curves.items()}), flush=True)

    out = ROOT / "results" / "related_work.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v.get(str(len(pool))) for k, v in curves.items()}, indent=1))


if __name__ == "__main__":
    main()
