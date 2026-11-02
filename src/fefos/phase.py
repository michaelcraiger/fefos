"""Convex hulls of multicomponent systems, self-consistent with a set of entries.

Used to place a predicted formation energy relative to every known phase
(distance to the hull, as in screening and in the stability benchmark of
Bartel et al., npj Comput. Mater. 6, 97, 2020). Needs scipy (``pip install
fefos[screen]``).

The hull energy at composition ``c`` is the lowest energy of any mixture of
known phases with that overall composition:

    E_hull(c) = min sum_k w_k E_k   s.t.  sum_k w_k x_k = c,   w >= 0,

over every entry ``k`` (atomic fractions ``x_k``, formation energy per atom
``E_k``; pure elements at zero) whose elements are a subset of those in ``c``.

Note: Materials Project's ``equilibrium_reaction_energy_per_atom`` is not
consistent with its formation energies for a sizeable fraction of stable
compounds in the 2026-09-28 release, so decomposition energies are recomputed
here instead.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class HullCalculator:
    """Hull energies from a list of ``(composition, formation_energy_per_atom)`` entries."""

    def __init__(self, entries: Iterable[tuple[Mapping[str, float], float]]):
        self._by_system: dict[str, list[tuple[dict[str, float], float]]] = defaultdict(list)
        for comp, energy in entries:
            total = sum(comp.values())
            frac = {el: n / total for el, n in comp.items() if n > 0}
            self._by_system["-".join(sorted(frac))].append((frac, float(energy)))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "HullCalculator":
        """From a frame with ``composition`` (``El:n;El:n``) and ``formation_energy_per_atom``."""
        from .data import parse_composition_column

        return cls(
            (parse_composition_column(c), e)
            for c, e in zip(frame.composition, frame.formation_energy_per_atom)
        )

    def hull_energy(self, composition: Mapping[str, float], exclude_self: bool = False) -> float:
        """Hull energy (eV/atom) at ``composition``.

        With ``exclude_self=True`` every entry with exactly this composition is
        left out, giving the hull of all *other* phases; the decomposition
        energy of a compound is then ``E_f - hull_energy(c, exclude_self=True)``.
        """
        from scipy.optimize import linprog

        els = sorted(el for el, n in composition.items() if n > 0)
        total = sum(composition[el] for el in els)
        target = np.array([composition[el] / total for el in els])
        columns, energies = [], []
        for k in range(1, len(els) + 1):
            for sub in combinations(els, k):
                if k == 1:
                    columns.append([1.0 if el == sub[0] else 0.0 for el in els])
                    energies.append(0.0)
                for frac, energy in self._by_system.get("-".join(sub), []):
                    x = [frac.get(el, 0.0) for el in els]
                    if exclude_self and np.allclose(x, target, atol=1e-6):
                        continue
                    columns.append(x)
                    energies.append(energy)
        res = linprog(
            np.array(energies),
            A_eq=np.array(columns).T,
            b_eq=target,
            bounds=(0, None),
            method="highs",
        )
        if not res.success:  # pragma: no cover - elements always make the problem feasible
            raise RuntimeError(res.message)
        return float(res.fun)

    def decomposition_energy(
        self, composition: Mapping[str, float], formation_energy: float
    ) -> float:
        """Distance to the hull of all other phases (negative = stable)."""
        return formation_energy - self.hull_energy(composition, exclude_self=True)


def mp_entries(version: str, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Every non-deprecated MP entry as ``material_id, composition, formation_energy_per_atom``."""
    path = Path(cache_dir) / f"entries_{version}.parquet" if cache_dir else None
    if path is not None and path.exists():
        return pd.read_parquet(path)
    from .data import fetch_mp_summary

    df = fetch_mp_summary(version)
    out = pd.DataFrame(
        {
            "material_id": df.material_id,
            "composition": [
                ";".join(f"{e}:{n:g}" for e, n in sorted(c.items())) for c in df.composition
            ],
            "formation_energy_per_atom": df.formation_energy_per_atom,
        }
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    return out
