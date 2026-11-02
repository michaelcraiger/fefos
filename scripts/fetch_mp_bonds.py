"""Download per-material bond lengths and bond-valence oxidation states from MP.

    python scripts/fetch_mp_bonds.py [--version 2026-09-28] [--data data/mp_2026-09-28]

Reads the public MP ``bonds`` and ``oxidation-states`` collections (AWS Open
Data, no API key) for the materials in ``binaries.csv`` and ``compounds.csv.gz``
and writes ``bonds.csv.gz``: one row per (material, cation element) with the
mean cation-anion bond length, the number of such bonds, and MP's bond-valence
oxidation state of that element.
"""

import argparse
from pathlib import Path

from fefos.data import DEFAULT_MP_VERSION, fetch_mp_bonds, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_MP_VERSION)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    data = Path(args.data or f"data/mp_{args.version}")
    binaries, compounds = load_dataset(data)
    anion_of = dict(zip(binaries.material_id, binaries.anion))
    anion_of.update(zip(compounds.material_id, compounds.anion))
    bonds = fetch_mp_bonds(anion_of, args.version)
    bonds.to_csv(data / "bonds.csv.gz", index=False, compression="gzip")
    print(
        f"{len(bonds)} (material, cation) bond records for {bonds.material_id.nunique()} materials"
    )


if __name__ == "__main__":
    main()
