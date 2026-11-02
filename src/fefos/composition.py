"""Composition parsing and the cation/anion view of a compound used by FEFOS."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Union

from .elements import ANION_VALENCE, NONMETALS, most_electronegative

CompositionLike = Union[str, Mapping[str, float]]

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\d+\.?\d*|\.\d+)|(\()|(\))")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a chemical formula such as ``"Dy(BiO2)3"`` into element amounts."""
    stack: list[defaultdict[str, float]] = [defaultdict(float)]
    last: dict[str, float] | str | None = None
    pos = 0
    for match in _TOKEN.finditer(formula.replace(" ", "")):
        if match.start() != pos:
            raise ValueError(f"Cannot parse formula {formula!r}")
        pos = match.end()
        element, number, open_paren, close_paren = match.groups()
        if element:
            stack[-1][element] += 1.0
            last = element
        elif number:
            amount = float(number)
            if isinstance(last, str):
                stack[-1][last] += amount - 1.0
            elif isinstance(last, dict):
                for el, n in last.items():
                    stack[-1][el] += n * (amount - 1.0)
            else:
                raise ValueError(f"Unexpected number in formula {formula!r}")
            last = None
        elif open_paren:
            stack.append(defaultdict(float))
            last = None
        elif close_paren:
            if len(stack) == 1:
                raise ValueError(f"Unbalanced parentheses in {formula!r}")
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] += n
            last = dict(group)
    if pos != len(formula.replace(" ", "")) or len(stack) != 1:
        raise ValueError(f"Cannot parse formula {formula!r}")
    return {el: n for el, n in stack[0].items() if n > 0}


def as_composition(composition: CompositionLike) -> dict[str, float]:
    if isinstance(composition, str):
        return parse_formula(composition)
    return {str(el): float(n) for el, n in composition.items() if float(n) > 0}


@dataclass(frozen=True)
class IonicComposition:
    """A compound viewed as cations M_i sharing a single anion X.

    Attributes
    ----------
    anion:
        The anion element X (by default the most electronegative element).
    valence:
        Nominal valence of X, used to convert composition into a mean cation
        oxidation state.
    cations:
        Mapping of cation element to its fraction of all cations (sums to 1).
    anion_per_cation:
        n_X / sum(n_M).
    """

    anion: str
    valence: int
    cations: dict[str, float]
    anion_per_cation: float

    @property
    def mean_oxidation_state(self) -> float:
        """Average formal oxidation state of the cations, ``valence * n_X / n_M``."""
        return self.valence * self.anion_per_cation

    @property
    def atoms_per_cation(self) -> float:
        return 1.0 + self.anion_per_cation

    @property
    def n_cations(self) -> int:
        return len(self.cations)

    @property
    def has_nonmetal_cation(self) -> bool:
        """True for polyanion / mixed-anion compounds (e.g. phosphates, oxyfluorides)."""
        return any(el in NONMETALS for el in self.cations)

    @classmethod
    def from_composition(
        cls,
        composition: CompositionLike,
        anion: str | None = None,
        valence: int | None = None,
    ) -> "IonicComposition":
        comp = as_composition(composition)
        if len(comp) < 2:
            raise ValueError("A compound needs at least one cation and one anion")
        anion = anion or most_electronegative(comp)
        if anion not in comp:
            raise ValueError(f"Anion {anion} not present in {comp}")
        if valence is None:
            if anion not in ANION_VALENCE:
                raise ValueError(
                    f"No default valence for anion {anion}; pass `valence=` explicitly"
                )
            valence = ANION_VALENCE[anion]
        n_cat = sum(n for el, n in comp.items() if el != anion)
        cations = {el: n / n_cat for el, n in sorted(comp.items()) if el != anion}
        return cls(anion, valence, cations, comp[anion] / n_cat)
