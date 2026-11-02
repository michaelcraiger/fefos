"""Screen hypothetical ternary oxides by predicted distance to the convex hull.

    python scripts/screen_oxides.py   # needs pymatgen (pip install -e ".[structure]")

Candidates: every pair of metal cations A, B at cation ratios 1:1, 1:2, 2:1,
1:3, 3:1, with the oxygen content fixed by charge balance using each element's
common oxidation states (pymatgen). Compositions already in Materials Project
are set aside as a calibration set. Each candidate gets

* dEf from FEFOS (linear) plus the oxide bond-length correction, and
* the hull energy at that composition from every MP phase in the A-B-O
  system (elements, intermetallics, binary oxides and known A-B-O compounds;
  for compositions already in MP, the hull of all *other* phases),

and is ranked by the predicted decomposition energy dHd = dEf - E_hull.
Negative dHd means FEFOS predicts the composition to be more stable than any
mixture of known phases. Writes ``results/screening.csv.gz``.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from fefos.cli import load_bond_data, load_model
from fefos.data import DEFAULT_MP_VERSION, load_dataset, parse_composition_column
from fefos.elements import NONMETALS
from fefos.phase import HullCalculator, mp_entries
from fefos.structure import predict_with_bonds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / f"mp_{DEFAULT_MP_VERSION}"
EXCLUDE = NONMETALS | {"Tc", "Pm", "Po", "At", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"}
RATIOS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)]
TOXIC = {"Tl", "Hg", "Cd", "Be", "Os", "Pb"}
PRECIOUS = {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au", "Re"}
THRESHOLD = -0.10  # eV/atom; see the calibration in results/screening_summary.json


def training_ranges(model, table, compounds, results):
    """1st-99th percentile of the bond descriptors over near-hull ionic oxides."""
    from fefos.structure import bond_features

    res = results[
        (results.anion == "O")
        & (results.mean_oxidation_state >= 1)
        & ~results.nonmetal_cation
        & (results.energy_above_hull < 0.1)
    ]
    comp = compounds.set_index("material_id").composition
    phi, size = [], []
    for r in res.itertuples(index=False):
        c = parse_composition_column(comp[r.material_id])
        states = {k: float(v) for k, v in (x.split(":") for x in r.oxidation_states.split(";"))}
        if all(table.has(el, "O") for el in states):
            cats = {el: n for el, n in c.items() if el != "O"}
            tot = sum(cats.values())
            f = bond_features(table, "O", {k: v / tot for k, v in cats.items()}, states)
            phi.append(f.ionic_potential_spread)
            size.append(f.size_mismatch)
    return {
        "ionic_potential_spread": np.percentile(phi, [1, 99]),
        "size_mismatch": np.percentile(size, [1, 99]),
    }


def candidates(elements):
    states = {el: [q for q in Element(el).common_oxidation_states if q > 0] for el in elements}
    for a, b in combinations(sorted(elements), 2):
        for (na, nb), qa, qb in product(RATIOS, states[a], states[b]):
            o = Fraction(na * qa + nb * qb, 2)
            k = o.denominator
            comp = Composition({a: na * k, b: nb * k, "O": o.numerator}).reduced_composition
            yield a, b, comp.reduced_formula, comp, {a: qa, b: qb}


def main() -> None:
    binaries, compounds = load_dataset(DATA)
    model = load_model(DATA)
    table, correction = load_bond_data(DATA)
    oxides = binaries[binaries.anion == "O"]
    metals = sorted(
        el
        for el in set(oxides.element) - EXCLUDE
        if (el, "O") in model.hulls and table.has(el, "O")
    )

    # every known A-B-O ternary (lowest polymorph per composition)
    tern = compounds[(compounds.anion == "O") & (compounds.n_cations == 2)]
    known, n_tern = {}, defaultdict(int)
    for r in tern.itertuples(index=False):
        comp = Composition(parse_composition_column(r.composition))
        pair = tuple(sorted(el.symbol for el in comp.elements if el.symbol != "O"))
        known[(pair, comp.reduced_formula)] = r
        n_tern[pair] += 1
    hulls = HullCalculator.from_frame(
        mp_entries(DEFAULT_MP_VERSION, ROOT / "benchmarks" / ".cache")
    )

    rows, seen = [], set()
    for a, b, formula, comp, oxi in candidates(metals):
        if (a, b, formula) in seen:
            continue
        seen.add((a, b, formula))
        amounts = dict(comp.get_el_amt_dict())
        r = predict_with_bonds(model, table, correction, amounts)
        k = known.get(((a, b), formula))
        # for compositions already in MP, compare with the hull of all *other* phases
        e_hull = hulls.hull_energy(amounts, exclude_self=k is not None)
        rows.append(
            {
                "formula": formula,
                "A": a,
                "B": b,
                "mean_oxidation_state": round(r.prediction.mean_oxidation_state, 4),
                "assumed_oxidation_states": f"{a}:{oxi[a]};{b}:{oxi[b]}",
                "fefos_oxidation_states": ";".join(
                    f"{el}:{v:.2f}" for el, v in r.prediction.oxidation_states.items()
                ),
                "fefos": round(r.prediction.formation_energy, 4),
                "fefos_bonds": round(r.formation_energy, 4),
                "size_mismatch": round(r.features.size_mismatch, 4),
                "ionic_potential_spread": round(r.features.ionic_potential_spread, 4),
                "hull_energy": round(e_hull, 4),
                "dhd_pred": round(r.formation_energy - e_hull, 4),
                "in_mp": k is not None,
                "mp_material_id": None if k is None else k.material_id,
                "mp_formation_energy": None if k is None else k.formation_energy_per_atom,
                "mp_dhd": None if k is None else round(k.formation_energy_per_atom - e_hull, 4),
                "n_known_ternaries": n_tern[(a, b)],
            }
        )
    print(f"{len(metals)} metals, {len(rows)} candidate compositions")

    out = pd.DataFrame(rows).sort_values("dhd_pred")
    assumed = out.assumed_oxidation_states.map(
        lambda s: dict((k, float(v)) for k, v in (x.split(":") for x in s.split(";")))
    )
    fefos_os = out.fefos_oxidation_states.map(
        lambda s: dict((k, float(v)) for k, v in (x.split(":") for x in s.split(";")))
    )
    out["charge_consistent"] = [
        all(abs(a[e] - f[e]) <= 0.25 for e in a) for a, f in zip(assumed, fefos_os)
    ]
    ranges = training_ranges(
        model, table, compounds, pd.read_csv(ROOT / "results" / "benchmark_predictions.csv.gz")
    )
    out["in_domain"] = np.ones(len(out), bool)
    for col, (lo, hi) in ranges.items():
        out["in_domain"] &= out[col].between(lo, hi)
    out["toxic"] = [bool({a, b} & TOXIC) for a, b in zip(out.A, out.B)]
    out["precious"] = [bool({a, b} & PRECIOUS) for a, b in zip(out.A, out.B)]
    out["shortlist"] = (
        ~out.in_mp & out.charge_consistent & out.in_domain & (out.dhd_pred < THRESHOLD)
    )
    out.to_csv(ROOT / "results" / "screening.csv.gz", index=False, compression="gzip")
    new = out[~out.in_mp]
    print(f"{len(out)} candidates, {out.in_mp.sum()} already in MP, {len(new)} new")
    print(f"new candidates predicted below the MP hull: {(new.dhd_pred < 0).sum()}")
    print(
        f"charge consistent: {new.charge_consistent.sum()}, in domain: {(new.charge_consistent & new.in_domain).sum()}"
    )
    short = out[out.shortlist]
    print(f"shortlist (consistent, in domain, dHd < {THRESHOLD}): {len(short)}")
    print(
        short.head(40)[
            [
                "formula",
                "fefos_bonds",
                "hull_energy",
                "dhd_pred",
                "fefos_oxidation_states",
                "n_known_ternaries",
                "toxic",
                "precious",
            ]
        ].to_string()
    )


if __name__ == "__main__":
    main()
