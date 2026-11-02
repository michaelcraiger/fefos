"""The FEFOS model: formation energies of multi-cation compounds from binary hulls.

A compound with cations M_1..M_n (fractions c_i) and one anion X has a mean
cation oxidation state ``q``. Every cation starts at the reference state ``q``
(as in the paper, where ABO3 is built from A2O3 and B2O3), with reference
energy read off its M–X convex hull. Charge-neutral shifts of the individual
oxidation states, priced by each element's energy-vs-oxidation-state curve,
then give the mixing energy dD and the formation energy (Eq. 6)::

    dE_f = sum_i c_i dE_f(M_i X_{q/v}) + dD.

Three curve/merging modes are available:

``"linear"`` (default)
    The curve *is* the convex hull (piecewise linear). Charge-neutral merging is
    solved exactly; the result equals the energy of the most stable mixture of
    binary compounds, and the optimum gives each cation's oxidation state.
``"quadratic"``
    Eq. 1 parabolas fitted (least squares, per cation) to the hull within
    ``window`` of the reference, merged with concentration weights.
``"paper"``
    The 2023 rule generalised to n cations: per-atom parabolas with the
    symmetric-minimum convention, unweighted merging and end-member correction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .composition import CompositionLike, IonicComposition, as_composition
from .curves import ReferenceCurves, fit_reference_curves
from .elements import ANION_VALENCE, most_electronegative
from .hull import BinaryHull
from .mixing import mix

MODES = ("linear", "quadratic", "paper")


@dataclass(frozen=True)
class Prediction:
    """FEFOS prediction for one compound (energies in eV/atom)."""

    formula: str
    anion: str
    mode: str
    mean_oxidation_state: float
    cations: dict[str, float]
    formation_energy: float
    reference_energy: float
    """Concentration-weighted hull energies of each cation at the mean oxidation state."""
    delta_d: float
    """Mixing energy from charge transfer, ``formation_energy - reference_energy``."""
    oxidation_states: dict[str, float]
    """Predicted oxidation state of each cation."""


@dataclass
class FEFOS:
    """Formation Energies From Oxidation States.

    Parameters
    ----------
    hulls:
        Mapping ``(element, anion) -> BinaryHull``.
    mode:
        ``"linear"``, ``"quadratic"`` or ``"paper"`` (see module docstring).
    window:
        Oxidation-state range on each side of the reference over which the
        quadratic curves are fitted (ignored in ``"linear"`` mode).
    """

    hulls: dict[tuple[str, str], BinaryHull]
    mode: str = "linear"
    window: float = 2.0
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    # ------------------------------------------------------------------ build
    @classmethod
    def from_binary_entries(cls, entries: Iterable[dict], **kwargs) -> "FEFOS":
        """Build from binary-compound records.

        Each record needs ``composition`` (formula or element->amount mapping)
        and ``formation_energy_per_atom``; ``entry_id`` and ``formula`` are
        optional. Records whose anion (most electronegative element) has no
        default valence in :data:`fefos.elements.ANION_VALENCE` are skipped.
        """
        grouped: dict[tuple[str, str], list] = defaultdict(list)
        for rec in entries:
            comp = as_composition(rec["composition"])
            if len(comp) != 2 or most_electronegative(comp) not in ANION_VALENCE:
                continue
            ionic = IonicComposition.from_composition(comp)
            (element,) = ionic.cations
            grouped[(element, ionic.anion)].append(
                (
                    str(rec.get("entry_id", "")),
                    str(rec.get("formula", "")),
                    comp[element],
                    comp[ionic.anion],
                    float(rec["formation_energy_per_atom"]),
                )
            )
        hulls = {
            (el, an): BinaryHull.from_entries(el, an, ANION_VALENCE[an], rows)
            for (el, an), rows in grouped.items()
        }
        return cls(hulls, **kwargs)

    def with_mode(self, mode: str, **kwargs) -> "FEFOS":
        """Same hulls, different mode (or window)."""
        return FEFOS(self.hulls, mode=mode, window=kwargs.get("window", self.window))

    # ---------------------------------------------------------------- queries
    def _ionic(self, composition: CompositionLike, anion: str | None) -> IonicComposition:
        """Cation/anion view, taking a non-standard anion's valence from the hulls."""
        valence = None
        if anion is not None and anion not in ANION_VALENCE:
            valence = next((h.valence for (_, an), h in self.hulls.items() if an == anion), None)
        return IonicComposition.from_composition(composition, anion, valence)

    def can_represent(self, composition: CompositionLike, anion: str | None = None) -> bool:
        try:
            ionic = self._ionic(composition, anion)
        except ValueError:
            return False
        return ionic.n_cations >= 1 and all((el, ionic.anion) in self.hulls for el in ionic.cations)

    def curves(self, element: str, anion: str, q_ref: float) -> ReferenceCurves:
        """Oxidation/reduction parabolas of ``element`` around ``q_ref``."""
        method = "paper" if self.mode == "paper" else "least_squares"
        key = (element, anion, round(q_ref, 9), method, self.window)
        if key not in self._cache:
            self._cache[key] = fit_reference_curves(
                self.hulls[(element, anion)], q_ref, window=self.window, method=method
            )
        return self._cache[key]

    def predict(self, composition: CompositionLike, anion: str | None = None) -> Prediction:
        """Predict the formation energy (eV/atom) of a compound."""
        comp = as_composition(composition)
        ionic = self._ionic(comp, anion)
        missing = [el for el in ionic.cations if (el, ionic.anion) not in self.hulls]
        if missing:
            raise KeyError(f"No {'/'.join(missing)}–{ionic.anion} binary data available")
        q = ionic.mean_oxidation_state
        elements = list(ionic.cations)
        c = np.array([ionic.cations[el] for el in elements])
        e_ref_cation = np.array(
            [float(self.hulls[(el, ionic.anion)].energy_per_cation(q)) for el in elements]
        )
        e_ref = float(np.dot(c, e_ref_cation)) / ionic.atoms_per_cation

        if self.mode == "linear":
            energy, states = self.decompose(ionic)
        else:
            refs = [self.curves(el, ionic.anion, q) for el in elements]
            rule = "paper" if self.mode == "paper" else "thermodynamic"
            result = mix(c, [r.ox for r in refs], [r.red for r in refs], rule=rule)
            scale = 1.0 if self.mode == "paper" else 1.0 / ionic.atoms_per_cation
            energy = e_ref + result.delta_d * scale
            states = {el: q + d for el, d in zip(elements, result.shifts)}
        return Prediction(
            formula=_formula(comp),
            anion=ionic.anion,
            mode=self.mode,
            mean_oxidation_state=q,
            cations=dict(ionic.cations),
            formation_energy=energy,
            reference_energy=e_ref,
            delta_d=energy - e_ref,
            oxidation_states={el: float(v) for el, v in states.items()},
        )

    def decompose(self, ionic: IonicComposition) -> tuple[float, dict[str, float]]:
        """Exact charge-neutral merge of the hull curves ("linear" mode).

        Every hull is convex and piecewise linear in oxidation state, so
        sharing the anions among the cations is a fractional knapsack: hull
        segments are filled from the steepest (most stabilising) slope until
        the anion budget ``sum_i c_i q_i = q`` is spent. Returns the energy per
        atom and the oxidation state of each cation. The slope of the last
        segment filled is the anion chemical potential of the mixture.
        """
        budget = ionic.mean_oxidation_state
        segments = []
        for el, c in ionic.cations.items():
            slopes, widths = self.hulls[(el, ionic.anion)].segments()
            segments.extend((s, c * w, el, c) for s, w in zip(slopes, widths))
        segments.sort(key=lambda t: t[0])
        energy = 0.0
        states = dict.fromkeys(ionic.cations, 0.0)
        for slope, capacity, el, c in segments:
            if budget <= 1e-15 or slope >= 0:
                break
            take = min(capacity, budget)
            energy += slope * take
            states[el] += take / c
            budget -= take
        return energy / ionic.atoms_per_cation, states


def _formula(comp: dict[str, float]) -> str:
    return "".join(el if abs(n - 1) < 1e-9 else f"{el}{n:g}" for el, n in comp.items())
