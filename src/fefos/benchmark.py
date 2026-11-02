"""Benchmark FEFOS against Materials Project formation energies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .composition import IonicComposition
from .data import binary_records, parse_composition_column
from .legacy import PaperFEFOS
from .model import FEFOS

PAPER_OXIDATION_STATES = (2.0, 3.0, 4.0)


def run_benchmark(
    binaries: pd.DataFrame,
    compounds: pd.DataFrame,
    window: float = 2.0,
    progress: bool = False,
) -> pd.DataFrame:
    """Predict every representable compound with each FEFOS mode and the baselines.

    Columns of the returned frame (energies in eV/atom):

    * ``mp`` – Materials Project formation energy (target)
    * ``fefos_linear`` / ``fefos_quadratic`` / ``fefos_paper`` – FEFOS modes
    * ``reference_average`` – Eq. 6 with dD = 0 (no charge transfer)
    * ``fefos_2023`` – published 2023 coefficients (paper scope only, else NaN)
    """
    base = FEFOS.from_binary_entries(binary_records(binaries), window=window)
    models = {m: base.with_mode(m) for m in ("linear", "quadratic", "paper")}
    paper_2023 = PaperFEFOS.load()
    rows = []
    for i, row in enumerate(compounds.itertuples(index=False)):
        if progress and i % 10000 == 0:
            print(f"  {i}/{len(compounds)}", flush=True)
        comp = parse_composition_column(row.composition)
        if not base.can_represent(comp, row.anion):
            continue
        preds = {m: model.predict(comp, row.anion) for m, model in models.items()}
        ionic = IonicComposition.from_composition(comp, row.anion)
        q = ionic.mean_oxidation_state
        paper_scope = (
            row.anion == "O"
            and ionic.n_cations == 2
            and any(abs(q - k) < 1e-6 for k in PAPER_OXIDATION_STATES)
        )
        fefos_2023 = np.nan
        if paper_scope and paper_2023.can_represent(comp):
            fefos_2023 = paper_2023.predict(comp, snap=False)["formation_energy"]
        lin = preds["linear"]
        rows.append(
            {
                "material_id": row.material_id,
                "formula": row.formula,
                "anion": row.anion,
                "n_cations": ionic.n_cations,
                "mean_oxidation_state": round(q, 6),
                "paper_scope": paper_scope,
                "nonmetal_cation": bool(row.nonmetal_cation),
                "is_stable": bool(row.is_stable),
                "theoretical": bool(row.theoretical),
                "energy_above_hull": row.energy_above_hull,
                "mp": row.formation_energy_per_atom,
                "fefos_linear": round(lin.formation_energy, 6),
                "fefos_quadratic": round(preds["quadratic"].formation_energy, 6),
                "fefos_paper": round(preds["paper"].formation_energy, 6),
                "reference_average": round(lin.reference_energy, 6),
                "fefos_2023": round(fefos_2023, 6),
                "oxidation_states": ";".join(
                    f"{el}:{v:.3f}" for el, v in lin.oxidation_states.items()
                ),
            }
        )
    return pd.DataFrame(rows)


MODELS = ("fefos_linear", "fefos_quadratic", "fefos_paper", "reference_average")


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    rmse: float
    median_ae: float
    r2: float
    bias: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def metrics(pred, target) -> Metrics:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(target)
    err = pred[mask] - target[mask]
    if err.size == 0:
        return Metrics(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((target[mask] - target[mask].mean()) ** 2))
    return Metrics(
        int(err.size),
        float(np.mean(np.abs(err))),
        float(np.sqrt(np.mean(err**2))),
        float(np.median(np.abs(err))),
        1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        float(np.mean(err)),
    )


def summarise(results: pd.DataFrame, by: list[str] | None = None, models=MODELS) -> pd.DataFrame:
    """Error metrics of each model, optionally grouped by result columns."""
    groups = [((), results)] if not by else results.groupby(by)
    out = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in models:
            met = metrics(g[m], g["mp"])
            if met.n == 0:
                continue
            out.append({**dict(zip(by or [], key)), "model": m, **met.as_dict()})
    return pd.DataFrame(out)
