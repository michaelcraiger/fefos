"""Element data used by FEFOS: electronegativities and anion valences."""

from __future__ import annotations

# Pauling electronegativities (as tabulated in pymatgen). Noble gases without a
# Pauling value are omitted; they can never be chosen as the anion.
# fmt: off
PAULING_ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.2, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.9, "P": 2.19, "S": 2.58,
    "Cl": 3.16, "K": 0.82, "Ca": 1.0, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.9, "Zn": 1.65, "Ga": 1.81,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.0, "Rb": 0.82, "Sr": 0.95,
    "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Tc": 1.9, "Ru": 2.2, "Rh": 2.28,
    "Pd": 2.2, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1,
    "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.1, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.2, "Gd": 1.2, "Tb": 1.1, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3, "Ta": 1.5,
    "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.2, "Pt": 2.28, "Au": 2.54, "Hg": 2.0,
    "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0, "At": 2.2, "Rn": 2.2, "Fr": 0.7,
    "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
}
# fmt: on

# Nominal (absolute) valence of each supported anion. The valence only sets the
# scale of the oxidation-state axis: for a compound with n_X anions per n_M
# cations the mean cation oxidation state is ``valence * n_X / n_M``.
ANION_VALENCE: dict[str, int] = {
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "H": 1,
    "O": 2,
    "S": 2,
    "Se": 2,
    "Te": 2,
    "N": 3,
    "P": 3,
    "C": 4,
}

# Elements that, when they appear alongside a more electronegative anion, act as
# the centre of a polyanion (e.g. P in phosphates, S in sulfates) or as a second
# anion (e.g. F in an oxyfluoride) rather than as a simple metal cation.
NONMETALS: frozenset[str] = frozenset(
    "H B C N O F Si P S Cl As Se Br Te I He Ne Ar Kr Xe Rn At".split()
)


def electronegativity(symbol: str) -> float:
    """Pauling electronegativity, or ``0.0`` for elements without one."""
    return PAULING_ELECTRONEGATIVITY.get(symbol, 0.0)


def most_electronegative(symbols) -> str:
    """Return the most electronegative element of ``symbols``."""
    return max(symbols, key=electronegativity)
