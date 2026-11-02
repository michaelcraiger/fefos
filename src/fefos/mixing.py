"""Charge-neutral merging of oxidation/reduction curves for any number of cations.

This generalises Eqs. 2-5 of the FEFOS paper from two cations to ``n``.

Each cation ``i`` (fraction ``c_i`` of all cations) shifts its oxidation state
by ``d_i`` from the common reference state. Charge neutrality (Eq. 2) requires

    sum_i c_i d_i = 0.

For two cations the paper writes ``d_A = x z`` and ``d_B = -x (1 - z)``, the
unique solution of this constraint along one direction. Each cation is given a
direction ``s_i`` (+1 oxidise, -1 reduce) and the curves are merged as

    dC = min_{s, d}  sum_i w_i f_i^{s_i}(|d_i|)   subject to  sum_i c_i d_i = 0.

Two weightings are supported:

``rule="paper"`` (``w_i = 1``)
    The 2023 rule: curve *arguments* are scaled by concentration but values are
    summed unweighted. Each curve can then lower the energy on its own even
    when its element is infinitely dilute, so the end-member correction (Eq. 5)
    subtracts those self-energies ``m_i = min_x f_i(x)``::

        dD = min_s [ dC_s - sum_i (1 - c_i) m_i^{s_i} ],

    which reduces exactly to Eq. 5 for two cations and vanishes for every pure
    end member.

``rule="thermodynamic"`` (``w_i = c_i``)
    Curves are per-cation energies weighted by concentration, i.e. the energy
    of a lever-rule mixture. No end-member correction is needed. With exact
    (piecewise-linear) hull curves this is the convex hull of the binaries.

For a fixed direction pattern the problem is a separable convex quadratic
programme with one linear constraint, solved to machine precision by
bisection on the Lagrange multiplier (the anion chemical potential).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .curves import Parabola

_A_MIN = 1e-7  # tiny curvature so straight-line curves have a unique optimum
_N_BISECT = 80
_X_CAP = 50.0


@dataclass(frozen=True)
class MixingResult:
    delta_d: float
    """Mixing energy relative to the reference average (Eq. 5)."""
    delta_c: float
    """Minimum of the merged curves before any end-member correction (Eq. 3)."""
    shifts: np.ndarray
    """Signed oxidation-state shift of each cation at the optimum."""


def _patterns(n: int) -> np.ndarray:
    """All +/-1 direction patterns with at least one oxidation and one reduction."""
    pats = np.array(list(product((1.0, -1.0), repeat=n)))
    keep = (pats > 0).any(axis=1) & (pats < 0).any(axis=1)
    return pats[keep]


def _solve(c, s, a, b, upper):
    """Row-wise minimum of sum(a u^2 + b u) s.t. sum(c s u) = 0, 0 <= u <= upper."""
    a = np.maximum(a, _A_MIN)
    cs = c * s
    two_a = 2.0 * a
    span = np.max((two_a * upper + np.abs(b)) / c, axis=1, keepdims=True) + 1.0
    lo, hi = -span, span
    for _ in range(_N_BISECT):
        mid = 0.5 * (lo + hi)
        u = np.clip((mid * cs - b) / two_a, 0.0, upper)
        below = np.sum(cs * u, axis=1, keepdims=True) < 0
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    u = np.clip((0.5 * (lo + hi) * cs - b) / two_a, 0.0, upper)
    return u, np.sum(a * u * u + b * u, axis=1)


def mix(
    concentrations: Sequence[float],
    ox: Sequence[Parabola],
    red: Sequence[Parabola],
    rule: str = "paper",
) -> MixingResult:
    """Minimise the charge-neutral combination of oxidation/reduction curves.

    Parameters
    ----------
    concentrations:
        Cation fractions ``c_i`` (summing to one).
    ox, red:
        Oxidation and reduction parabola of each cation; ``Parabola.x_max``
        bounds the shift in that direction.
    rule:
        ``"paper"`` (unweighted sum + end-member correction, the 2023 rule) or
        ``"thermodynamic"`` (concentration-weighted sum).
    """
    if rule not in ("paper", "thermodynamic"):
        raise ValueError(f"Unknown mixing rule {rule!r}")
    c = np.asarray(concentrations, dtype=float)
    n = c.size
    if n < 2:
        return MixingResult(0.0, 0.0, np.zeros(n))
    pats = _patterns(n)
    is_ox = pats > 0
    ox_p = np.array([[p.a, p.b, p.x_max] for p in ox])
    red_p = np.array([[p.a, p.b, p.x_max] for p in red])
    w = np.ones(n) if rule == "paper" else c
    a = np.where(is_ox, ox_p[:, 0], red_p[:, 0]) * w
    b = np.where(is_ox, ox_p[:, 1], red_p[:, 1]) * w
    upper = np.minimum(np.where(is_ox, ox_p[:, 2], red_p[:, 2]), _X_CAP)
    u, delta_c = _solve(np.broadcast_to(c, pats.shape), pats, a, b, upper)

    delta_d = delta_c
    if rule == "paper":
        m_ox = np.array([_capped(p).minimum()[1] for p in ox])
        m_red = np.array([_capped(p).minimum()[1] for p in red])
        delta_d = delta_c - np.sum((1.0 - c) * np.where(is_ox, m_ox, m_red), axis=1)
    k = int(np.argmin(delta_d))
    return MixingResult(float(delta_d[k]), float(delta_c[k]), pats[k] * u[k])


def _capped(p: Parabola) -> Parabola:
    return Parabola(p.a, p.b, min(p.x_max, _X_CAP))
