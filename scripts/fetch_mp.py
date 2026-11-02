"""Download the Materials Project snapshot used by the benchmark.

    python scripts/fetch_mp.py [--version 2026-09-28] [--out data/mp_2026-09-28]

Reads the public MP summary parquet from AWS Open Data (no API key) and writes
``binaries.csv`` (hull inputs) and ``compounds.csv.gz`` (benchmark targets).
"""

import argparse

from fefos.data import (
    DEFAULT_MP_VERSION,
    fetch_mp_summary,
    save_dataset,
    split_binaries_and_compounds,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_MP_VERSION)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"data/mp_{args.version}"
    summary = fetch_mp_summary(args.version)
    binaries, compounds = split_binaries_and_compounds(summary)
    save_dataset(binaries, compounds, out)
    print(
        f"{len(summary)} MP entries -> {len(binaries)} binaries, {len(compounds)} compounds in {out}"
    )


if __name__ == "__main__":
    main()
