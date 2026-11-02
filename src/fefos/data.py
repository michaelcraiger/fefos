"""Materials Project data from the public AWS Open Data bucket (no API key needed).

``fetch_mp_summary`` reads only the columns FEFOS needs from the MP summary
parquet snapshot; ``split_binaries_and_compounds`` turns that into

* the binary M–X compounds used to build convex hulls, and
* the multi-cation single-anion compounds used as the benchmark, keeping the
  lowest-energy polymorph of each composition (FEFOS is composition-only).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .composition import IonicComposition
from .elements import ANION_VALENCE, most_electronegative

MP_BUCKET = "materialsproject-build"
MP_SUMMARY_PREFIX = "collections/summary"
DEFAULT_MP_VERSION = "2026-09-28"

_COLUMNS = [
    "material_id",
    "formula_pretty",
    "composition_reduced",
    "nelements",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "theoretical",
    "deprecated",
]


def fetch_mp_summary(version: str = DEFAULT_MP_VERSION) -> pd.DataFrame:
    """Download the slim MP summary table (≈280k rows) from AWS Open Data."""
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    prefix = f"{MP_BUCKET}/{MP_SUMMARY_PREFIX}/version={version}"
    files = [
        f.path for f in s3.get_file_info(pafs.FileSelector(prefix)) if f.path.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"No parquet files under s3://{prefix}")
    tables = [pq.ParquetFile(p, filesystem=s3).read(columns=_COLUMNS) for p in files]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    df["composition"] = df.pop("composition_reduced").apply(
        lambda pairs: {str(el): float(n) for el, n in pairs}
    )
    df = df[~df["deprecated"] & df["formation_energy_per_atom"].notna()].drop(columns="deprecated")
    return df.reset_index(drop=True)


def fetch_mp_structures(material_ids, version: str = DEFAULT_MP_VERSION) -> dict:
    """Crystal structures (pymatgen dict form) of the given MP materials."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    prefix = f"{MP_BUCKET}/{MP_SUMMARY_PREFIX}/version={version}"
    ids = pa.array(list(material_ids))
    out = {}
    for info in s3.get_file_info(pafs.FileSelector(prefix)):
        if not info.path.endswith(".parquet"):
            continue
        pf = pq.ParquetFile(info.path, filesystem=s3)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["material_id", "structure"])
            t = t.filter(pc.is_in(t["material_id"], value_set=ids))
            out.update(zip(t["material_id"].to_pylist(), t["structure"].to_pylist()))
    return out


_BOND_KEY = re.compile(r"^([A-Z][a-z]?)-([A-Z][a-z]?)$")


def cation_anion_bonds(bond_types, anion: str) -> dict[str, tuple[float, int]]:
    """Mean bond length and bond count from each cation to ``anion``.

    ``bond_types`` is MP's mapping (or list of pairs) ``"A-B" -> [lengths]``;
    keys that are not two element symbols, and bonds not involving exactly one
    ``anion``, are ignored.
    """
    items = bond_types.items() if isinstance(bond_types, Mapping) else (bond_types or [])
    out: dict[str, tuple[float, int]] = {}
    for key, lengths in items:
        m = _BOND_KEY.match(key)
        if not m or not lengths:
            continue
        a, b = m.groups()
        if (a == anion) == (b == anion):
            continue
        out[b if a == anion else a] = (float(np.mean(lengths)), len(lengths))
    return out


def fetch_mp_bonds(anion_of: dict[str, str], version: str = DEFAULT_MP_VERSION) -> pd.DataFrame:
    """Mean cation-anion bond lengths and bond-valence oxidation states from MP.

    ``anion_of`` maps material id -> anion element; only those materials are
    kept. Returns one row per (material, cation element) with columns
    ``material_id, element, anion, bond_length, n_bonds, bv_oxidation_state``.
    Bond lengths come from MP's ``bonds`` collection (CrystalNN neighbours);
    oxidation states from its ``oxidation-states`` collection (bond valence).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    ids = pa.array(list(anion_of))

    def read(collection: str, columns: list[str]):
        prefix = f"{MP_BUCKET}/collections/{collection}/version={version}"
        files = [
            f.path
            for f in s3.get_file_info(pafs.FileSelector(prefix))
            if f.path.endswith(".parquet")
        ]
        tables = []
        for path in files:
            pf = pq.ParquetFile(path, filesystem=s3)
            for g in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(g, columns=columns)
                mask = pc.is_in(t["material_id"], value_set=ids)
                tables.append(t.filter(mask))
        return tables

    bond_rows = []
    for t in read("bonds", ["material_id", "bond_types"]):
        for mid, types in zip(t["material_id"].to_pylist(), t["bond_types"].to_pylist()):
            anion = anion_of[mid]
            for cation, (mean, n) in cation_anion_bonds(types, anion).items():
                bond_rows.append((mid, cation, anion, mean, n))
    bonds = pd.DataFrame(
        bond_rows, columns=["material_id", "element", "anion", "bond_length", "n_bonds"]
    )
    ox_rows = []
    for t in read("oxidation-states", ["material_id", "average_oxidation_states"]):
        for mid, avg in zip(
            t["material_id"].to_pylist(), t["average_oxidation_states"].to_pylist()
        ):
            for el, v in avg or []:
                ox_rows.append((mid, el, v))
    ox = pd.DataFrame(ox_rows, columns=["material_id", "element", "bv_oxidation_state"])
    out = bonds.merge(ox, on=["material_id", "element"], how="left")
    out["bond_length"] = out["bond_length"].round(4)
    return out.sort_values(["material_id", "element"]).reset_index(drop=True)


def split_binaries_and_compounds(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(binaries, compounds)`` tables from an MP summary table."""
    df = summary.copy()
    df["anion"] = df["composition"].apply(most_electronegative)
    df = df[df["anion"].isin(ANION_VALENCE)]

    bin_rows = df[df["nelements"] == 2]
    binaries = pd.DataFrame(
        {
            "material_id": bin_rows["material_id"],
            "formula": bin_rows["formula_pretty"],
            "element": [
                next(el for el in c if el != a)
                for c, a in zip(bin_rows["composition"], bin_rows["anion"])
            ],
            "anion": bin_rows["anion"],
            "n_element": [
                next(n for el, n in c.items() if el != a)
                for c, a in zip(bin_rows["composition"], bin_rows["anion"])
            ],
            "n_anion": [c[a] for c, a in zip(bin_rows["composition"], bin_rows["anion"])],
            "formation_energy_per_atom": bin_rows["formation_energy_per_atom"].round(6),
            "energy_above_hull": bin_rows["energy_above_hull"].round(6),
            "theoretical": bin_rows["theoretical"],
        }
    ).sort_values(["anion", "element", "formation_energy_per_atom"])

    multi = df[df["nelements"] >= 3].sort_values("formation_energy_per_atom")
    multi = multi.drop_duplicates("formula_pretty", keep="first")
    records = []
    for row in multi.itertuples(index=False):
        ionic = IonicComposition.from_composition(row.composition, row.anion)
        records.append(
            {
                "material_id": row.material_id,
                "formula": row.formula_pretty,
                "anion": ionic.anion,
                "n_cations": ionic.n_cations,
                "composition": ";".join(f"{el}:{n:g}" for el, n in row.composition.items()),
                "mean_oxidation_state": round(ionic.mean_oxidation_state, 6),
                "nonmetal_cation": ionic.has_nonmetal_cation,
                "formation_energy_per_atom": round(row.formation_energy_per_atom, 6),
                "energy_above_hull": round(row.energy_above_hull, 6),
                "is_stable": bool(row.is_stable),
                "theoretical": bool(row.theoretical),
            }
        )
    compounds = pd.DataFrame.from_records(records).sort_values(["anion", "formula"])
    return binaries.reset_index(drop=True), compounds.reset_index(drop=True)


def parse_composition_column(value: str) -> dict[str, float]:
    """Inverse of the ``El:n;El:n`` encoding used in the compounds table."""
    out = {}
    for part in value.split(";"):
        el, n = part.split(":")
        out[el] = float(n)
    return out


def binary_records(binaries: pd.DataFrame):
    """Yield records for :meth:`fefos.FEFOS.from_binary_entries`."""
    for row in binaries.itertuples(index=False):
        yield {
            "entry_id": row.material_id,
            "formula": row.formula,
            "composition": {row.element: row.n_element, row.anion: row.n_anion},
            "formation_energy_per_atom": row.formation_energy_per_atom,
        }


def load_dataset(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(directory)
    binaries = pd.read_csv(directory / "binaries.csv")
    compounds = pd.read_csv(directory / "compounds.csv.gz")
    return binaries, compounds


def save_dataset(binaries: pd.DataFrame, compounds: pd.DataFrame, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binaries.to_csv(directory / "binaries.csv", index=False)
    compounds.to_csv(directory / "compounds.csv.gz", index=False, compression="gzip")


__all__ = [
    "DEFAULT_MP_VERSION",
    "binary_records",
    "cation_anion_bonds",
    "fetch_mp_bonds",
    "fetch_mp_structures",
    "fetch_mp_summary",
    "load_dataset",
    "parse_composition_column",
    "save_dataset",
    "split_binaries_and_compounds",
]
