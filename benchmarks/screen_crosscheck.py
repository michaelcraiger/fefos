"""Cross-check the oxide screen with FEFOS + a learned residual.

    python benchmarks/run_related.py          # held-out hulls
    python scripts/screen_oxides.py           # FEFOS screen
    python benchmarks/screen_crosscheck.py    # -> results/screening_crosscheck.{csv.gz,json}

FEFOS with the bond correction is the fast, interpretable screen. Gradient
boosting on Magpie descriptors trained on FEFOS's residual ranks stability far
better (benchmarks/run_related.py), so it is used here as a second opinion:

1. Held out: train the residual model without the test chemical systems and
   measure the hit rate of each flag (FEFOS screen, residual model, both) on
   held-out two-cation oxides.
2. Full: train on every compound and score the screening candidates.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import composition_key, magpie_features  # noqa: E402
from run_related import CACHE, DATA, ROOT, apply_correction, build_tables, fit_correction, hgb  # noqa: E402

from fefos.cli import load_model  # noqa: E402
from fefos.data import parse_composition_column  # noqa: E402

EXTRA = ["fefos", "reference", "mean_os", "phi", "size", "n_cations"]
THRESHOLD = -0.10
STABLE_TOL = -0.001


def rates(dhd: pd.Series, mask) -> dict:
    mask = np.asarray(mask)
    return {
        "n": int(mask.sum()),
        "stable": round(float((dhd[mask] < 0).mean()), 3) if mask.any() else None,
        "near_hull": round(float((dhd[mask] < 0.05).mean()), 3) if mask.any() else None,
    }


def main() -> None:
    multi, bins = build_tables()
    magpie = pd.read_parquet(CACHE / "magpie.parquet")

    def X(df):
        return np.hstack([magpie.loc[df.key].to_numpy(), df[EXTRA].to_numpy(float)])

    hulls = pd.read_csv(ROOT / "results" / "related_work_test_hulls.csv.gz").set_index(
        "material_id"
    )
    test = multi[multi.fold == 0].copy()
    pool = multi[multi.fold != 0]
    test["dhd"] = hulls.dhd.reindex(test.material_id).to_numpy()
    test["hull_others"] = hulls.hull_others.reindex(test.material_id).to_numpy()
    train = pd.concat([bins, pool], ignore_index=True)
    model = hgb(X(train), (train.mp - train.fefos).to_numpy(), 0)
    test["dhd_bonds"] = apply_correction(fit_correction(pool), test) - test.hull_others
    test["dhd_ml"] = test.fefos + model.predict(X(test)) - test.hull_others
    ox = test[(test.anion == "O") & (test.n_cations == 2)]
    a, b = ox.dhd_bonds < THRESHOLD, ox.dhd_ml < STABLE_TOL
    calib = {
        "all held-out two-cation oxides": rates(ox.dhd, np.ones(len(ox), bool)),
        "FEFOS screen (bond correction < -0.10)": rates(ox.dhd, a),
        "residual model predicts stable": rates(ox.dhd, b),
        "both": rates(ox.dhd, a & b),
    }

    # full model -> score the screening candidates
    full = pd.concat([bins, multi], ignore_index=True)
    model = hgb(X(full), (full.mp - full.fefos).to_numpy(), 0)
    screen = pd.read_csv(ROOT / "results" / "screening.csv.gz")
    cand = screen[~screen.in_mp].copy()
    fefos_model = load_model(DATA)
    comps = [
        dict(parse_composition_column(";".join(f"{e}:{n}" for e, n in _amounts(f).items())))
        for f in cand.formula
    ]
    preds = [fefos_model.predict(c, "O") for c in comps]
    cand["key"] = [composition_key(c) for c in comps]
    cand["reference"] = [p.reference_energy for p in preds]
    cand["n_cations"] = 2
    cand["phi"] = cand.ionic_potential_spread
    cand["size"] = cand.size_mismatch
    cand["mean_os"] = cand.mean_oxidation_state
    feats = magpie_features(comps).to_numpy()
    Xc = np.hstack([feats, cand[EXTRA].to_numpy(float)])
    cand["residual_model_dhd"] = (cand.fefos + model.predict(Xc) - cand.hull_energy).round(4)
    cand["consensus"] = cand.shortlist & (cand.residual_model_dhd < STABLE_TOL)
    cols = ["formula", "dhd_pred", "residual_model_dhd", "shortlist", "consensus"]
    cand[cols].sort_values("dhd_pred").to_csv(
        ROOT / "results" / "screening_crosscheck.csv.gz", index=False, compression="gzip"
    )
    out = {
        "held_out_calibration": calib,
        "n_shortlist": int(cand.shortlist.sum()),
        "n_consensus": int(cand.consensus.sum()),
        "n_residual_model_stable_new": int((cand.residual_model_dhd < STABLE_TOL).sum()),
        "consensus": cand[cand.consensus]
        .sort_values("dhd_pred")[cols[:3]]
        .round(4)
        .to_dict("records"),
    }
    (ROOT / "results" / "screening_crosscheck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "consensus"}, indent=1))
    print(pd.DataFrame(out["consensus"]).to_string())


def _amounts(formula: str) -> dict:
    from pymatgen.core import Composition

    return Composition(formula).get_el_amt_dict()


if __name__ == "__main__":
    main()
