"""Quadratic oxidation/reduction curves (Eq. 1 of the FEFOS paper).

For an element M referenced at oxidation state ``q_ref`` FEFOS describes how
the formation energy responds to oxidation (``q_ref + x``) and reduction
(``q_ref - x``) with two parabolas through the origin::

    f_ox(x)  = a1 x^2 + a2 x,       f_red(x) = a1' x^2 + a2' x,      x >= 0.

In the 2023 paper the coefficients were fitted to Materials Project entries
around the integer reference states MO, M2O3 and MO2, with several hand-tuned
corrections. Here they are fitted automatically to the thermodynamic convex
hull of the M–X system (:mod:`fefos.hull`) around *any* reference state, so
the same procedure works for any anion and any mean oxidation state.

Two fitting conventions are provided:

``"least_squares"``
    Least-squares fit (non-negative curvature) to the hull energy *per
    cation*, sampled uniformly over ``0 < x <= window``. Energies per cation
    combine linearly, so these curves can be merged thermodynamically.
``"paper"``
    The 2023 convention: energies *per atom*; if the hull dips below the
    reference in a direction, the symmetric parabola through that minimum is
    used (so the curve's minimum marks the preferred oxidation state),
    otherwise a least-squares fit through the lower hull of the points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hull import BinaryHull, lower_convex_hull


@dataclass(frozen=True)
class Parabola:
    """``f(x) = a x^2 + b x`` on ``0 <= x <= x_max``."""

    a: float
    b: float
    x_max: float = np.inf

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return self.a * x * x + self.b * x

    def minimum(self) -> tuple[float, float]:
        """``(x*, f(x*))`` minimising the parabola on ``[0, x_max]``."""
        candidates = [0.0]
        if np.isfinite(self.x_max):
            candidates.append(self.x_max)
        if self.a > 0:
            x_star = -self.b / (2 * self.a)
            if 0 < x_star < self.x_max:
                candidates.append(x_star)
        values = [float(self(x)) for x in candidates]
        i = int(np.argmin(values))
        return candidates[i], values[i]

    @classmethod
    def fit(cls, x, y, x_max: float = np.inf) -> "Parabola":
        """Least-squares fit through the origin with non-negative curvature."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]
        if x.size == 0:
            return cls(0.0, 0.0, 0.0)
        if np.unique(x).size >= 2:
            design = np.column_stack([x * x, x])
            (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
            if a >= 0:
                return cls(float(a), float(b), x_max)
        # concave (or a single point): fall back to the best straight line
        return cls(0.0, float(np.dot(x, y) / np.dot(x, x)), x_max)

    @classmethod
    def through_minimum(cls, x0: float, y0: float, x_max: float = np.inf) -> "Parabola":
        """Parabola through the origin with its vertex at ``(x0, y0)``."""
        return cls(-y0 / x0**2, 2.0 * y0 / x0, x_max)


@dataclass(frozen=True)
class ReferenceCurves:
    """Reference energy plus oxidation/reduction parabolas for one element."""

    element: str
    anion: str
    reference_oxidation_state: float
    reference_energy: float
    """Hull formation energy of M X_{q_ref/v}, per atom (``"paper"``) or per cation."""
    ox: Parabola
    red: Parabola
    per_atom: bool


def fit_reference_curves(
    hull: BinaryHull,
    q_ref: float,
    window: float = 2.0,
    method: str = "least_squares",
    n_samples: int = 81,
) -> ReferenceCurves:
    """Fit Eq. 1 parabolas to the hull around oxidation state ``q_ref``.

    Oxidation is fitted on ``0 < x <= window`` and reduction on
    ``0 < x <= min(window, q_ref)`` (an element cannot be reduced below the
    metal).
    """
    if method not in ("least_squares", "paper"):
        raise ValueError(f"Unknown fitting method {method!r}")
    per_atom = method == "paper"
    energy = hull.energy_per_atom if per_atom else hull.energy_per_cation
    e_ref = float(energy(q_ref))
    parabolas = []
    for direction in (+1, -1):
        x_max = window if direction > 0 else min(window, q_ref)
        if x_max <= 1e-9:
            parabolas.append(Parabola(0.0, 0.0, 0.0))
            continue
        if method == "least_squares":
            x = np.linspace(0.0, x_max, n_samples)[1:]
            y = energy(q_ref + direction * x) - e_ref
            parabolas.append(Parabola.fit(x, y, x_max=x_max))
            continue
        # paper convention: hull vertices within the window plus the window edge
        v = hull.vertices_q
        inside = (direction * (v - q_ref) > 1e-12) & (np.abs(v - q_ref) <= x_max + 1e-12)
        x = np.append(np.abs(v[inside] - q_ref), x_max)
        y = energy(q_ref + direction * x) - e_ref
        i = int(np.argmin(y))
        if y[i] < -1e-9 and x[i] < x_max - 1e-9:
            parabolas.append(Parabola.through_minimum(x[i], y[i], x_max))
        else:
            pts = lower_convex_hull([(0.0, 0.0), *zip(x, y)])
            parabolas.append(Parabola.fit([p[0] for p in pts], [p[1] for p in pts], x_max=x_max))
    ox, red = parabolas
    return ReferenceCurves(hull.element, hull.anion, float(q_ref), e_ref, ox, red, per_atom)
