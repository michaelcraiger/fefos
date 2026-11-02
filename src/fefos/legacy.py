"""Exact reproduction of the 2023 FEFOS paper predictions.

Uses the published parabola coefficients and reference oxides
(``data/paper_2023``) and the binary mixing rule of the original notebook
``formation_energy_mixer.ipynb`` (tag ``paper-2023``): two cations sharing an
integer reference oxidation state (2, 3 or 4), concentrations snapped to a
0.02 grid, a bounded scan variable ``x in [0, 10]`` and end-member subtraction
per oxidation direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .composition import CompositionLike, IonicComposition

PAPER_DIR = Path(__file__).resolve().parents[2] / "data" / "paper_2023"
_X_MAX = 10.0
_GRID = np.linspace(0.0, 1.0, 51)


def _data_dir() -> Path:
    if not PAPER_DIR.exists():
        raise FileNotFoundError(
            f"{PAPER_DIR} not found; the 2023 data ship with the repository, "
            "install FEFOS from a clone (pip install -e .)"
        )
    return PAPER_DIR


def paper_grid_index(value: float) -> int:
    """Nearest point of the paper's 51-point concentration grid (original tie-breaking)."""
    idx = int(np.searchsorted(_GRID, value, side="left"))
    if idx > 0 and (
        idx == len(_GRID) or math.fabs(value - _GRID[idx - 1]) < math.fabs(value - _GRID[idx])
    ):
        return idx - 1
    return idx


def snap_concentration(z_a: float, a: str, b: str, key_element: str | None = None) -> float:
    """Concentration of ``a`` as evaluated by the original notebook.

    The notebook stored each compound under the concentration of one element,
    rounded to two decimals, and looked up the prediction on a 0.02 grid
    indexed by the concentration of the alphabetically later element.
    """
    later, earlier = (b, a) if b > a else (a, b)
    z_later = z_a if later == a else 1.0 - z_a
    if key_element is None or key_element == later:
        idx = paper_grid_index(round(z_later, 2))
    else:
        idx = paper_grid_index(1.0 - round(1.0 - z_later, 2))
    z_later = float(_GRID[idx])
    return z_later if later == a else 1.0 - z_later


def _min_quadratic(A: float, B: float, x_max: float = _X_MAX) -> float:
    """Global minimum of ``A x^2 + B x`` on ``[0, x_max]``."""
    cands = [0.0, x_max]
    if A > 0 and 0 < -B / (2 * A) < x_max:
        cands.append(-B / (2 * A))
    return min(A * x * x + B * x for x in cands)


@dataclass
class PaperFEFOS:
    """FEFOS as published in 2023 (binary oxides with mean OS 2, 3 or 4 only)."""

    ox: dict[tuple[str, int], tuple[float, float]]
    red: dict[tuple[str, int], tuple[float, float]]
    reference_energy: dict[tuple[str, int], float]

    @classmethod
    def load(cls, directory: Path | None = None) -> "PaperFEFOS":
        d = directory or _data_dir()
        coeffs = pd.read_csv(d / "parabola_coeffs.csv", skipinitialspace=True)
        refs = pd.read_csv(d / "reference_oxides.csv", skipinitialspace=True)
        ox, red, eref = {}, {}, {}
        for row in coeffs.itertuples(index=False):
            key = _key(row.oxide)
            ox[key] = (row[1], row[2])
            red[key] = (row[3], row[4])
        for row in refs.itertuples(index=False):
            eref[_key(row.oxide)] = row[2]
        return cls(ox, red, eref)

    def can_represent(self, composition: CompositionLike) -> bool:
        try:
            ionic = IonicComposition.from_composition(composition, "O")
        except ValueError:
            return False
        q = ionic.mean_oxidation_state
        if ionic.n_cations != 2 or abs(q - round(q)) > 1e-6 or round(q) not in (2, 3, 4):
            return False
        return all(
            (el, round(q)) in self.ox and (el, round(q)) in self.reference_energy
            for el in ionic.cations
        )

    def delta_d(
        self,
        a: str,
        b: str,
        q: int,
        z_a: float,
        snap: bool = True,
        key_element: str | None = None,
    ) -> float:
        """Mixing energy dD of A_{z_a} B_{1-z_a} O_{q/2} (paper Eqs. 3-5).

        With ``snap=True`` the concentration is snapped to the paper's grid
        exactly as the original notebook did (see :func:`snap_concentration`).
        """
        if snap:
            z_a = snap_concentration(z_a, a, b, key_element)
        o_a, r_b = self.ox[(a, q)], self.red[(b, q)]
        o_b, r_a = self.ox[(b, q)], self.red[(a, q)]

        def branch(o, r, c_ox):
            # A oxidises by x(1-c_ox), B reduces by x*c_ox (charge neutral)
            def dc(c):
                A = o[0] * (1 - c) ** 2 + r[0] * c * c
                B = o[1] * (1 - c) + r[1] * c
                return _min_quadratic(A, B)

            return dc(c_ox) - (1 - c_ox) * dc(0.0) - c_ox * dc(1.0)

        return min(branch(o_a, r_b, z_a), branch(o_b, r_a, 1.0 - z_a))

    def predict(
        self, composition: CompositionLike, snap: bool = True, key_element: str | None = None
    ) -> dict:
        ionic = IonicComposition.from_composition(composition, "O")
        if not self.can_represent(composition):
            raise ValueError("Composition outside the scope of the 2023 FEFOS model")
        q = int(round(ionic.mean_oxidation_state))
        (a, z_a), (b, z_b) = ionic.cations.items()
        e_ref = z_a * self.reference_energy[(a, q)] + z_b * self.reference_energy[(b, q)]
        dd = self.delta_d(a, b, q, z_a, snap=snap, key_element=key_element)
        return {"formation_energy": e_ref + dd, "reference_energy": e_ref, "delta_d": dd}


def _key(oxide_name: str) -> tuple[str, int]:
    ionic = IonicComposition.from_composition(oxide_name.strip(), "O")
    (el,) = ionic.cations
    return el, int(round(ionic.mean_oxidation_state))


def published_predictions() -> pd.DataFrame:
    """The predictions table released with the 2023 paper."""
    return pd.read_csv(_data_dir() / "fefos_predictions.csv")


def benchmark_set() -> pd.DataFrame:
    """The 2023 benchmark compounds (MP 2022 energies, concentration keys, coordination flag)."""
    return pd.read_csv(_data_dir() / "benchmark_set.csv")


def reproduce_published() -> pd.DataFrame:
    """Recompute every published prediction; returns both columns side by side."""
    model = PaperFEFOS.load()
    pub = published_predictions()
    keys = benchmark_set().drop_duplicates(["mp_id", "formula"]).set_index(["mp_id", "formula"])
    out = []
    for row in pub.itertuples(index=False):
        key_el = keys["key_element"].get((row[1], row[0]))
        pred = model.predict(row[0], key_element=key_el)
        out.append(
            {
                "formula": row[0],
                "mp_id": row[1],
                "mp_formation_energy": row[2],
                "published_prediction": row[3],
                "reproduced_prediction": pred["formation_energy"],
                "reference_energy": pred["reference_energy"],
                "delta_d": pred["delta_d"],
                "coordination_match": bool(row[8]),
            }
        )
    return pd.DataFrame(out)
