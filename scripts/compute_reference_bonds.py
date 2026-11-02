"""Bond lengths, coordination and bond-valence charges of the reference binaries.

    python scripts/compute_reference_bonds.py [--data data/mp_2026-09-28] [--workers 6]

For the lowest-energy polymorph of every binary composition within 0.1 eV/atom
of the hull, downloads the structure from the MP summary (AWS Open Data) and
runs CrystalNN + bond-valence analysis (pymatgen). Writes ``binary_bonds.csv``.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fefos.data import DEFAULT_MP_VERSION, fetch_mp_structures, load_dataset
from fefos.elements import ANION_VALENCE
from fefos.structure import structure_bonds


def reference_candidates(binaries: pd.DataFrame, max_e_above_hull: float = 0.1) -> pd.DataFrame:
    b = binaries.assign(q=(binaries.n_anion / binaries.n_element).round(6))
    ground = b.sort_values("formation_energy_per_atom").drop_duplicates(["element", "anion", "q"])
    return ground[ground.energy_above_hull <= max_e_above_hull]


def _run(args):
    mid, anion, structure = args
    try:
        return mid, structure_bonds(structure, anion)
    except Exception as exc:  # noqa: BLE001 - keep going, report the failure
        return mid, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=f"data/mp_{DEFAULT_MP_VERSION}")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    data = Path(args.data)
    binaries, _ = load_dataset(data)
    cand = reference_candidates(binaries)
    structures = fetch_mp_structures(cand.material_id.tolist())
    anion_of = dict(zip(cand.material_id, cand.anion))
    jobs = [(mid, anion_of[mid], s) for mid, s in structures.items()]
    with Pool(args.workers) as pool:
        results = pool.map(_run, jobs, chunksize=8)
    rows, failed = [], []
    meta = cand.set_index("material_id")
    for mid, res in results:
        if isinstance(res, str):
            failed.append((mid, res))
            continue
        m = meta.loc[mid]
        for el, r in res.items():
            rows.append(
                {
                    "material_id": mid,
                    "formula": m.formula,
                    "element": el,
                    "anion": m.anion,
                    "formal_oxidation_state": round(
                        ANION_VALENCE[m.anion] * m.n_anion / m.n_element, 6
                    ),
                    "energy_above_hull": m.energy_above_hull,
                    **r,
                }
            )
    out = pd.DataFrame(rows).sort_values(["anion", "element", "formal_oxidation_state"])
    for col in ("bond_length", "coordination", "bv_oxidation_state"):
        out[col] = out[col].round(4)
    out.to_csv(data / "binary_bonds.csv", index=False)
    print(
        f"{len(out)} reference binaries; {len(failed)} failed; "
        f"{np.isnan(out.bv_oxidation_state).mean():.1%} without a bond-valence charge"
    )
    for mid, err in failed[:10]:
        print("  failed", mid, err[:80])


if __name__ == "__main__":
    main()
