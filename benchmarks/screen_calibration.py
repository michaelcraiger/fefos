"""Out-of-sample hit rate of the oxide screen.

    python benchmarks/run_related.py      # writes results/related_work_test_hulls.csv.gz
    python benchmarks/screen_calibration.py

The bond-length correction is fitted without the held-out chemical systems and
applied to their two-cation oxides. For each threshold on the predicted
distance to the hull, reports how many flagged compounds are actually stable
in MP (and within 50 meV/atom of the hull). Writes results/screening_calibration.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_related import ROOT, apply_correction, build_tables, fit_correction  # noqa: E402


def main() -> None:
    multi, _ = build_tables()
    hulls = pd.read_csv(ROOT / "results" / "related_work_test_hulls.csv.gz").set_index(
        "material_id"
    )
    test = multi[multi.fold == 0].copy()
    pool = multi[multi.fold != 0]
    test["dhd"] = hulls.dhd.reindex(test.material_id).to_numpy()
    test["hull_others"] = hulls.hull_others.reindex(test.material_id).to_numpy()
    test["pred"] = apply_correction(fit_correction(pool), test)
    test["dhd_pred"] = test.pred - test.hull_others
    out = {}
    for name, g in {
        "two-cation oxides": test[(test.anion == "O") & (test.n_cations == 2)],
        "all held-out ionic compounds": test,
    }.items():
        rows = [
            {
                "threshold": None,
                "n": int(len(g)),
                "stable": round(float((g.dhd < 0).mean()), 3),
                "near_hull": round(float((g.dhd < 0.05).mean()), 3),
            }
        ]
        for thr in (0.0, -0.05, -0.10):
            m = g.dhd_pred < thr
            rows.append(
                {
                    "threshold": thr,
                    "n": int(m.sum()),
                    "stable": round(float((g.dhd[m] < 0).mean()), 3) if m.any() else np.nan,
                    "near_hull": round(float((g.dhd[m] < 0.05).mean()), 3) if m.any() else np.nan,
                }
            )
        out[name] = rows
    (ROOT / "results" / "screening_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
