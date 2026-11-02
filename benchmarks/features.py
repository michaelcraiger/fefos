"""Magpie composition features (Ward et al., npj Comput. Mater. 2, 16028, 2016).

    python benchmarks/features.py   # -> benchmarks/.cache/magpie.parquet

The 132 features of matminer's ``ElementProperty.from_preset("magpie")``
(22 elemental properties x min, max, range, mean, avg_dev, mode), computed
vectorised from matminer's Magpie element table, which is much faster than
featurising compound by compound.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData

from fefos.data import DEFAULT_MP_VERSION, load_dataset, parse_composition_column

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "benchmarks" / ".cache"


def composition_key(comp: dict) -> str:
    total = sum(comp.values())
    return ";".join(f"{el}:{n / total:.6f}" for el, n in sorted(comp.items()))


def magpie_features(compositions: list[dict]) -> pd.DataFrame:
    preset = ElementProperty.from_preset("magpie")
    props = preset.features
    data = MagpieData()
    elements = sorted({el for c in compositions for el in c})
    table = np.array(
        [[data.get_elemental_property(_el(e), p) for p in props] for e in elements], float
    )
    col_mean = np.nanmean(table, axis=0)
    table = np.where(np.isnan(table), col_mean, table)  # impute like impute_nan=True
    pos = {e: i for i, e in enumerate(elements)}
    out = np.zeros((len(compositions), 6 * len(props)))
    for n, comp in enumerate(compositions):
        els = list(comp)
        w = np.array([comp[e] for e in els], float)
        w = w / w.sum()
        v = table[[pos[e] for e in els]]
        mean = w @ v
        out[n, 0::6] = v.min(0)
        out[n, 1::6] = v.max(0)
        out[n, 2::6] = v.max(0) - v.min(0)
        out[n, 3::6] = mean
        out[n, 4::6] = w @ np.abs(v - mean)
        out[n, 5::6] = v[np.argmax(w)]  # mode: property of the most abundant element
    cols = [
        f"{s} {p}"
        for p in props
        for s in ("minimum", "maximum", "range", "mean", "avg_dev", "mode")
    ]
    return pd.DataFrame(out, columns=cols)


def _el(symbol):
    from pymatgen.core import Element

    return Element(symbol)


def main() -> None:
    binaries, compounds = load_dataset(ROOT / "data" / f"mp_{DEFAULT_MP_VERSION}")
    comps = [parse_composition_column(s) for s in compounds.composition]
    comps += [{r.element: r.n_element, r.anion: r.n_anion} for r in binaries.itertuples()]
    keys = [composition_key(c) for c in comps]
    uniq = dict(zip(keys, comps))
    feats = magpie_features(list(uniq.values()))
    feats.index = list(uniq)
    CACHE.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(CACHE / "magpie.parquet")
    print(feats.shape)


if __name__ == "__main__":
    main()
