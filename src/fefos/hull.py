"""One-dimensional convex hulls of binary M–X systems on an oxidation-state axis.

For an element M and anion X (valence v) a binary compound M_a X_b sits at the
formal oxidation state ``q = v * b / a`` and has a formation energy per cation

    e(q) = E_f(per atom) * (1 + b / a).

Working per cation makes mixing linear (the lever rule), so the thermodynamic
convex hull of the M–X system is the lower convex hull of ``(q, e)`` together
with the pure metal at ``(0, 0)``. Beyond the most anion-rich stable compound
the hull is flat: extra anion is released as the pure element at zero energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


def lower_convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex hull of 2-D points (Andrew's monotone chain), sorted by x."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    # keep the lowest y for duplicated x
    dedup: dict[float, float] = {}
    for x, y in pts:
        dedup[x] = min(y, dedup.get(x, np.inf))
    pts = sorted(dedup.items())
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # remove the middle point if it lies on or above the segment
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 1e-12:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


@dataclass(frozen=True)
class BinaryEntry:
    """A binary M_a X_b compound placed on the oxidation-state axis."""

    entry_id: str
    formula: str
    oxidation_state: float
    energy_per_cation: float
    formation_energy_per_atom: float


@dataclass
class BinaryHull:
    """Thermodynamic hull of an M–X binary system as a function of oxidation state."""

    element: str
    anion: str
    valence: int
    vertices_q: np.ndarray
    vertices_e: np.ndarray
    entries: list[BinaryEntry] = field(default_factory=list, repr=False)

    @classmethod
    def from_entries(
        cls,
        element: str,
        anion: str,
        valence: int,
        entries: Iterable[tuple[str, str, float, float, float]],
    ) -> "BinaryHull":
        """Build the hull from ``(id, formula, n_M, n_X, formation_energy_per_atom)``."""
        parsed: list[BinaryEntry] = []
        for entry_id, formula, n_m, n_x, ef in entries:
            r = n_x / n_m
            parsed.append(BinaryEntry(entry_id, formula, valence * r, ef * (1.0 + r), ef))
        pts = [(0.0, 0.0)] + [(p.oxidation_state, p.energy_per_cation) for p in parsed]
        hull = lower_convex_hull(pts)
        # Keep only the decreasing branch: anything past the minimum is unstable
        # with respect to the minimum-energy compound plus the pure anion.
        i_min = int(np.argmin([e for _, e in hull]))
        hull = hull[: i_min + 1]
        q = np.array([p[0] for p in hull])
        e = np.array([p[1] for p in hull])
        return cls(element, anion, valence, q, e, parsed)

    @property
    def max_stable_oxidation_state(self) -> float:
        """Oxidation state of the most anion-rich stable compound (0 if none)."""
        return float(self.vertices_q[-1])

    @property
    def has_stable_compound(self) -> bool:
        return len(self.vertices_q) > 1

    def energy_per_cation(self, q) -> np.ndarray:
        """Hull formation energy per cation at oxidation state(s) ``q``."""
        q = np.asarray(q, dtype=float)
        return np.interp(q, self.vertices_q, self.vertices_e, left=np.nan)

    def energy_per_atom(self, q) -> np.ndarray:
        """Hull formation energy per atom of composition M X_{q/v}."""
        q = np.asarray(q, dtype=float)
        return self.energy_per_cation(q) / (1.0 + q / self.valence)

    def e_above_hull(self, entry: BinaryEntry) -> float:
        """Energy above the hull of an entry, per atom."""
        e_hull = float(self.energy_per_atom(entry.oxidation_state))
        return entry.formation_energy_per_atom - e_hull

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Slopes (eV per unit oxidation state) and widths of the hull segments."""
        dq = np.diff(self.vertices_q)
        return np.diff(self.vertices_e) / dq, dq
