"""Assumed bond lengths: validate them and test them as an energy correction.

    python scripts/compute_reference_bonds.py   # reference binaries (pymatgen)
    python scripts/fetch_mp_bonds.py            # bond lengths of the benchmark compounds
    python scripts/run_benchmark.py             # FEFOS predictions and oxidation states
    python scripts/bond_length_analysis.py

Writes
* ``data/<mp>/bond_lengths.csv``     reference table d(M, X, charge)
* ``data/<mp>/bond_correction.csv``  per-anion correction fitted on near-hull compounds
* ``results/bond_lengths.json``      validation numbers and chart samples
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from fefos import IonicComposition
from fefos.data import DEFAULT_MP_VERSION, load_dataset, parse_composition_column
from fefos.structure import (
    CORRECTION_FEATURES,
    BondLengthCorrection,
    BondLengthTable,
    bond_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / f"mp_{DEFAULT_MP_VERSION}"
RES = ROOT / "results"
FOLDS = 5


def parse_states(s: str) -> dict[str, float]:
    return {k: float(v) for k, v in (p.split(":") for p in s.split(";"))}


def mae(a, b) -> float:
    return round(float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))), 4)


def main() -> None:
    _, compounds = load_dataset(DATA)
    ref = pd.read_csv(DATA / "binary_bonds.csv")
    table = BondLengthTable.from_reference_bonds(ref)
    table.to_frame().round(4).to_csv(DATA / "bond_lengths.csv", index=False)
    bonds = pd.read_csv(DATA / "bonds.csv.gz").set_index(["material_id", "element"])
    res = pd.read_csv(RES / "benchmark_predictions.csv.gz")
    res = res[(res.mean_oxidation_state >= 1) & ~res.nonmetal_cation]
    comp_of = compounds.set_index("material_id").composition
    agnostic = ref.dropna(subset=["bond_length"]).groupby(["element", "anion"]).bond_length.median()

    sites, rows = [], []
    for r in res.itertuples(index=False):
        ion = IonicComposition.from_composition(
            parse_composition_column(comp_of[r.material_id]), r.anion
        )
        if not all(table.has(el, r.anion) for el in ion.cations):
            continue
        states = parse_states(r.oxidation_states)
        f = bond_features(table, r.anion, ion.cations, states)
        rows.append(
            {
                "material_id": r.material_id,
                "formula": r.formula,
                "anion": r.anion,
                "n_cations": r.n_cations,
                "chemsys": "-".join(sorted(ion.cations)),
                "mp": r.mp,
                "fefos": r.fefos_linear,
                "energy_above_hull": r.energy_above_hull,
                "is_stable": r.is_stable,
                "theoretical": r.theoretical,
                **{k: getattr(f, k) for k in CORRECTION_FEATURES},
            }
        )
        for el in ion.cations:
            if (r.material_id, el) not in bonds.index:
                continue
            a = bonds.loc[(r.material_id, el)]
            if not np.isfinite(a.bv_oxidation_state):
                continue
            sites.append(
                {
                    "formula": r.formula,
                    "anion": r.anion,
                    "element": el,
                    "stable": r.is_stable,
                    "theoretical": r.theoretical,
                    "actual": a.bond_length,
                    "agnostic": agnostic.get((el, r.anion), np.nan),
                    "mean_os": table.assumed_bond_length(el, r.anion, r.mean_oxidation_state),
                    "fefos": f.bond_lengths[el],
                    "bv": table.assumed_bond_length(el, r.anion, a.bv_oxidation_state),
                    "q_fefos": states[el],
                    "q_mean": r.mean_oxidation_state,
                    "q_bv": a.bv_oxidation_state,
                }
            )

    # ---- A. assumed vs actual bond lengths
    S = pd.DataFrame(sites).dropna()
    variants = ["agnostic", "mean_os", "fefos", "bv"]
    part_a = {
        "n_sites": int(len(S)),
        "n_compounds": int(S.formula.nunique()),
        "subsets": {
            name: {"n": int(len(g)), **{v: mae(g[v], g.actual) for v in variants}}
            for name, g in [("all", S), ("stable", S[S.stable]), ("ICSD", S[~S.theoretical])]
        },
        "by_anion": {
            an: {"n": int(len(g)), **{v: mae(g[v], g.actual) for v in variants}}
            for an, g in S.groupby("anion")
        },
        "relative_mae_fefos": round(float(np.mean(np.abs(S.fefos - S.actual) / S.actual)), 4),
        "charge_deviation_fefos": mae(S.q_fefos, S.q_bv),
        "charge_deviation_mean_os": mae(S.q_mean, S.q_bv),
        "sample": S.sample(1500, random_state=0)[["formula", "element", "anion", "actual", "fefos"]]
        .round(3)
        .values.tolist(),
    }

    # ---- B. energy correction, validated out of sample
    B = pd.DataFrame(rows)
    B["residual"] = B.mp - B.fefos
    B["fold"] = [zlib.crc32(s.encode()) % FOLDS for s in B.chemsys]
    near = B.energy_above_hull < 0.1
    X = B[list(CORRECTION_FEATURES)].to_numpy()

    def corrected(train, test, use_features=True):
        feats = X if use_features else np.zeros_like(X)
        corr = BondLengthCorrection.fit(B.anion[train], feats[train], B.residual[train])
        c = np.array([corr.coefficients.get(a, np.zeros(3)) for a in B.anion[test]])
        return B.fefos[test].to_numpy() + c[:, 0] + np.sum(c[:, 1:] * feats[test], axis=1)

    subsets = {
        "all": np.ones(len(B), bool),
        "stable": B.is_stable.to_numpy(),
        "ICSD": ~B.theoretical.to_numpy(),
        "near hull": near.to_numpy(),
    }

    def scores(pred, mask):
        mp = B.mp.to_numpy()
        return {k: mae(pred[m & mask], mp[m & mask]) for k, m in subsets.items()}

    protocols = {}
    # 1. folds by chemical system
    p_off, p_bond = np.zeros(len(B)), np.zeros(len(B))
    for k in range(FOLDS):
        tr, te = (B.fold != k) & near, (B.fold == k).to_numpy()
        p_off[te] = corrected(tr, te, use_features=False)
        p_bond[te] = corrected(tr, te)
    every = np.ones(len(B), bool)
    protocols["chemical systems held out (5-fold)"] = {
        "FEFOS": scores(B.fefos.to_numpy(), every),
        "+ anion offset": scores(p_off, every),
        "+ offset and bond features": scores(p_bond, every),
    }
    # 2. train on two-cation compounds, test on three or more
    tr, te = (B.n_cations == 2) & near, (B.n_cations >= 3).to_numpy()
    full_off, full_bond = B.fefos.to_numpy().copy(), B.fefos.to_numpy().copy()
    full_off[te] = corrected(tr, te, use_features=False)
    full_bond[te] = corrected(tr, te)
    protocols["trained on 2 cations, tested on 3+"] = {
        "FEFOS": scores(B.fefos.to_numpy(), te),
        "+ anion offset": scores(full_off, te),
        "+ offset and bond features": scores(full_bond, te),
    }
    # 3. coefficients fitted on oxides, applied unchanged to every other anion
    tr, te = ((B.anion == "O") & near).to_numpy(), (B.anion != "O").to_numpy()
    xfer_off, xfer_bond = B.fefos.to_numpy().copy(), B.fefos.to_numpy().copy()
    for use, out_arr in ((False, xfer_off), (True, xfer_bond)):
        feats = X if use else np.zeros_like(X)
        c = BondLengthCorrection.fit(B.anion[tr], feats[tr], B.residual[tr]).coefficients["O"]
        out_arr[te] += c[0] + feats[te] @ c[1:]
    protocols["oxide coefficients applied to other anions"] = {
        "FEFOS": scores(B.fefos.to_numpy(), te),
        "+ anion offset": scores(xfer_off, te),
        "+ offset and bond features": scores(xfer_bond, te),
    }
    correlations = {
        f: {
            "all": round(float(np.corrcoef(B.residual, B[f])[0, 1]), 3),
            "near hull": round(float(np.corrcoef(B.residual[near], B[f][near])[0, 1]), 3),
        }
        for f in CORRECTION_FEATURES
    }
    final = BondLengthCorrection.fit(B.anion[near], X[near], B.residual[near])
    final.to_frame().round(5).to_csv(DATA / "bond_correction.csv", index=False)

    out = {
        "n_reference_binaries": int(ref.material_id.nunique()),
        "n_reference_pairs": len(table.points),
        "bond_lengths": part_a,
        "energy": {
            "n_compounds": int(len(B)),
            "correlations": correlations,
            "protocols": protocols,
            "coefficients": final.to_frame().round(4).to_dict("records"),
        },
    }
    (RES / "bond_lengths.json").write_text(json.dumps(out, default=float))
    print(
        json.dumps({k: v for k, v in out.items() if k != "bond_lengths"}, indent=1, default=float)[
            :2500
        ]
    )
    print(json.dumps({k: v for k, v in part_a.items() if k != "sample"}, default=float)[:1500])


if __name__ == "__main__":
    main()
