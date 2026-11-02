"""Assumed bond lengths: reference cation-anion distances for an element at a given charge.

The 2023 paper pointed to ionic size (tolerance factors, the tau descriptor) as
the natural next ingredient for FEFOS. This module provides it without using
the structure of the compound being predicted:

1. Every binary compound M_aX_b near its convex hull gives one reference point
   ``(q, d)``: the charge ``q`` of M (bond-valence charge, so Li is +1 in Li2O2;
   the formal ``v*b/a`` when no valid assignment exists) and its mean M-X bond
   length ``d`` (CrystalNN neighbours, as in Materials Project).
2. ``BondLengthTable.assumed_bond_length(M, X, q)`` interpolates those points
   linearly in ``q`` (clamped outside the reference range), so an element gets
   a characteristic bond length for any charge FEFOS assigns to it.
3. ``bond_features`` turns the assumed lengths of a compound's cations into
   descriptors: size mismatch and the spread of ionic potential ``q/d``
   (Cartier's ionic potential, an acid-base scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class BondLengthTable:
    """Reference M-X bond length as a function of the oxidation state of M."""

    points: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]

    @classmethod
    def from_reference_bonds(cls, frame: pd.DataFrame) -> "BondLengthTable":
        """Build the table from reference binary compounds.

        ``frame`` needs ``element, anion, bond_length, energy_above_hull`` and
        ``formal_oxidation_state``; an optional ``bv_oxidation_state`` column
        (bond-valence charge, e.g. +1 for Li in Li2O2 rather than the formal +2)
        is used wherever it is finite. Where several references share a charge
        the most stable one is kept.
        """
        df = frame.dropna(subset=["bond_length"]).copy()
        charge = df["formal_oxidation_state"].astype(float)
        if "bv_oxidation_state" in df:
            charge = df["bv_oxidation_state"].where(np.isfinite(df["bv_oxidation_state"]), charge)
        df["charge"] = charge.round(3)
        df = df[df["charge"] > 0]
        df = df.sort_values("energy_above_hull").drop_duplicates(["element", "anion", "charge"])
        points = {}
        for (el, an), g in df.sort_values("charge").groupby(["element", "anion"]):
            points[(el, an)] = (g["charge"].to_numpy(float), g["bond_length"].to_numpy(float))
        return cls(points)

    def has(self, element: str, anion: str) -> bool:
        return (element, anion) in self.points

    def assumed_bond_length(self, element: str, anion: str, q: float) -> float:
        """Bond length (Angstrom) of M at oxidation state ``q``, interpolated in ``q``."""
        qs, ds = self.points[(element, anion)]
        return float(np.interp(q, qs, ds))

    def to_frame(self) -> pd.DataFrame:
        rows = [
            (el, an, q, d)
            for (el, an), (qs, ds) in sorted(self.points.items())
            for q, d in zip(qs, ds)
        ]
        return pd.DataFrame(rows, columns=["element", "anion", "oxidation_state", "bond_length"])

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BondLengthTable":
        points = {}
        for (el, an), g in frame.sort_values("oxidation_state").groupby(["element", "anion"]):
            points[(el, an)] = (g.oxidation_state.to_numpy(float), g.bond_length.to_numpy(float))
        return cls(points)


@dataclass(frozen=True)
class BondFeatures:
    """Size and charge descriptors of a compound from assumed bond lengths."""

    bond_lengths: dict[str, float]
    """Assumed M-X bond length of each cation (Angstrom)."""
    ionic_potentials: dict[str, float]
    """q/d of each cation (e/Angstrom)."""
    size_mismatch: float
    """Concentration-weighted standard deviation of d divided by its mean."""
    ionic_potential_spread: float
    """Concentration-weighted standard deviation of q/d."""


def bond_features(
    table: BondLengthTable,
    anion: str,
    concentrations: Mapping[str, float],
    oxidation_states: Mapping[str, float],
) -> BondFeatures:
    """Descriptors for cations with the given fractions and oxidation states."""
    els = list(concentrations)
    c = np.array([concentrations[e] for e in els], dtype=float)
    q = np.array([oxidation_states[e] for e in els], dtype=float)
    d = np.array([table.assumed_bond_length(e, anion, qi) for e, qi in zip(els, q)])
    phi = q / d
    d_mean = float(c @ d)
    return BondFeatures(
        bond_lengths=dict(zip(els, d.tolist())),
        ionic_potentials=dict(zip(els, phi.tolist())),
        size_mismatch=float(np.sqrt(c @ (d - d_mean) ** 2) / d_mean),
        ionic_potential_spread=float(np.sqrt(c @ (phi - c @ phi) ** 2)),
    )


# --------------------------------------------------------------------------
# Reference bonds from crystal structures (needs pymatgen: pip install fefos[structure])
def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_clean(v) for v in d]
    return d


def structure_bonds(structure, anion: str) -> dict[str, dict[str, float]]:
    """Cation-anion bonds of one crystal structure.

    ``structure`` is a pymatgen ``Structure`` or its dict form. Neighbours are
    found with CrystalNN (as in Materials Project) and charges assigned with
    the bond-valence method. Returns, per cation element: mean ``bond_length``
    to the anion, mean ``coordination`` (anion neighbours per site),
    ``n_bonds`` and ``bv_oxidation_state`` (NaN when no valid assignment).
    """
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Structure

    if not isinstance(structure, Structure):
        structure = Structure.from_dict(_clean(structure))
    try:
        valences = BVAnalyzer().get_valences(structure)
    except ValueError:
        valences = [np.nan] * len(structure)
    cnn = CrystalNN()
    per_el: dict[str, dict[str, list]] = {}
    for i, site in enumerate(structure):
        el = site.specie.symbol
        if el == anion:
            continue
        d = [
            float(np.linalg.norm(nn["site"].coords - site.coords))
            for nn in cnn.get_nn_info(structure, i)
            if nn["site"].specie.symbol == anion
        ]
        rec = per_el.setdefault(el, {"lengths": [], "cn": [], "ox": []})
        rec["lengths"] += d
        rec["cn"].append(len(d))
        rec["ox"].append(valences[i])
    return {
        el: {
            "bond_length": float(np.mean(r["lengths"])) if r["lengths"] else np.nan,
            "coordination": float(np.mean(r["cn"])),
            "n_bonds": len(r["lengths"]),
            "bv_oxidation_state": float(np.mean(r["ox"])),
        }
        for el, r in per_el.items()
    }


# --------------------------------------------------------------------------
# Empirical correction: what the binary hull misses, from assumed bond lengths
CORRECTION_FEATURES = ("ionic_potential_spread", "size_mismatch")


@dataclass
class BondLengthCorrection:
    """Per-anion linear correction ``offset + b1*phi_spread + b2*size_mismatch``.

    FEFOS (linear mode) predicts the energy of the most stable mixture of
    binary compounds; real compounds are on average more stable than that.
    The shortfall grows with the size mismatch and the ionic-potential contrast
    of the cations, computed from assumed bond lengths at FEFOS's oxidation
    states. The coefficients are fitted per anion (they do not transfer between
    anions), so this is an empirical, data-fitted term on top of FEFOS.
    """

    coefficients: dict[str, np.ndarray]

    @classmethod
    def fit(cls, anions, features: np.ndarray, residuals, min_samples: int = 30):
        """Least-squares fit per anion; ``features`` has one column per feature."""
        anions = np.asarray(anions)
        X = np.column_stack([np.ones(len(anions)), np.asarray(features, float)])
        y = np.asarray(residuals, float)
        coefs = {}
        for an in np.unique(anions):
            m = anions == an
            if m.sum() >= min_samples:
                coefs[str(an)], *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        return cls(coefs)

    def __call__(self, anion: str, features: BondFeatures) -> float:
        """Energy correction (eV/atom); zero for anions without coefficients."""
        c = self.coefficients.get(anion)
        if c is None:
            return 0.0
        x = [1.0] + [getattr(features, f) for f in CORRECTION_FEATURES]
        return float(np.dot(c, x))

    def to_frame(self) -> pd.DataFrame:
        cols = ["offset", *CORRECTION_FEATURES]
        return pd.DataFrame(
            [[an, *c] for an, c in sorted(self.coefficients.items())], columns=["anion", *cols]
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "BondLengthCorrection":
        cols = ["offset", *CORRECTION_FEATURES]
        return cls({r.anion: frame.loc[i, cols].to_numpy(float) for i, r in frame.iterrows()})


@dataclass(frozen=True)
class BondPrediction:
    """FEFOS prediction plus assumed bond lengths and the bond-length correction."""

    prediction: "object"  # fefos.model.Prediction
    features: BondFeatures
    correction: float

    @property
    def formation_energy(self) -> float:
        """FEFOS (linear) formation energy plus the bond-length correction (eV/atom)."""
        return self.prediction.formation_energy + self.correction


def predict_with_bonds(
    model, table: BondLengthTable, correction: BondLengthCorrection, composition
):
    """Predict with FEFOS (linear mode) and add the bond-length correction.

    Assumed bond lengths are evaluated at the oxidation states FEFOS predicts.
    The correction was fitted to linear-mode energies, so ``model.mode`` must
    be ``"linear"``.
    """
    if model.mode != "linear":
        raise ValueError("the bond-length correction is fitted to FEFOS in 'linear' mode")
    p = model.predict(composition)
    f = bond_features(table, p.anion, p.cations, p.oxidation_states)
    return BondPrediction(p, f, correction(p.anion, f))
